"""Kiem toan doc lap moi con so se dua vao bang ket qua cuoi.

Nguyen tac: KHONG tin file tong hop. Moi con so trong bang duoc tinh LAI tu
nguon tho (checkpoint tung lan lap, tham so da luu, du lieu goc) bang mot
duong tinh viet lai o day (sap nhan, Bias/RMSE/Coverage, QLIKE/NLS...), roi
doi chieu voi file tong hop ma cac script sinh bang dang doc. Chi dung lai
thu vien o cho khong the tranh: bo loc tien / vong lap phuong sai / mat do.

Ly do can no: hai loi o tang gop (dao nhan, p_ss so voi 1,0) da lot qua sau
khi ket qua duoc coi la xong. Bao cao nay la thu bien "toi nghi dung" thanh
"da doi chieu".

Cach chay (tu thu muc `code thuat toan`):
    py scripts/tien_ich/kiem_toan_ket_qua.py              # toan bo
    py scripts/tien_ich/kiem_toan_ket_qua.py --phan sim   # chi mo phong
    py scripts/tien_ich/kiem_toan_ket_qua.py --phan buoc1
    py scripts/tien_ich/kiem_toan_ket_qua.py --phan buoc3
    py scripts/tien_ich/kiem_toan_ket_qua.py --spot 2     # tinh lai Brier tu dau
                                                          # cho 2 lan lap/kich ban
Ket qua: results/kiem_toan_ket_qua.txt
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from msgarchx.inference import model_spec                       # noqa: E402
from msgarchx.inference.param_spec import ModelConfig           # noqa: E402
from msgarchx.models.egarch import EGARCHParams, egarch_forward  # noqa: E402
from msgarchx.models.garch import GARCHParams, garch_forward     # noqa: E402
from msgarchx.models.ms_garch import MSGARCHXParams, ms_garch_forward_filter  # noqa: E402
from msgarchx.distributions import normal_logpdf, std_t_logpdf  # noqa: E402
from msgarchx.simulation.scenarios import default_scenarios     # noqa: E402
from msgarchx.simulation.run_egarch import (                    # noqa: E402
    default_egarch_scenarios, named_params_egarch,
)

RES = ROOT / "results"
TOL = 1e-8          # sai so cho phep khi doi chieu hai cach tinh cung mot dai luong
TOL_LOGLIK = 1e-6   # tong 1735 so hang, sai so lam tron float64 lon hon mot chut

DONG: list[str] = []
TONG = {"dat": 0, "hong": 0, "canh_bao": 0}
GRAD = 0            # so lan lap moi kich ban kiem tra gradient tai nghiem MLE (--grad)


def ghi(s: str = "") -> None:
    DONG.append(s)
    print(s)


def ket(ten: str, dat: bool, chi_tiet: str = "", canh_bao: bool = False) -> None:
    if canh_bao:
        TONG["canh_bao"] += 1
        nhan = "CANH BAO"
    elif dat:
        TONG["dat"] += 1
        nhan = "DAT"
    else:
        TONG["hong"] += 1
        nhan = "HONG"
    ghi(f"  [{nhan:8}] {ten}" + (f"  -- {chi_tiet}" if chi_tiet else ""))


def chenh(a, b) -> float:
    return abs(float(a) - float(b))


# ===========================================================================
# Phan 0: du lieu thuc
# ===========================================================================
def kiem_du_lieu() -> dict:
    ghi("=" * 78)
    ghi("PHAN 0 -- DU LIEU THUC (results/00_data.pt)")
    ghi("=" * 78)
    d = torch.load(RES / "00_data.pt", weights_only=False)
    n = int(d["r_all"].shape[0])
    n_train = int(d["n_train"])
    ket("tong quan sat = 2170", n == 2170, f"n={n}")
    ket("huan luyen/kiem dinh = 1735/435", (n_train, n - n_train) == (1735, 435),
        f"{n_train}/{n - n_train}")
    dates = d.get("dates_all")
    if dates is not None:
        dates = list(dates)
        ghi(f"  huan luyen: {str(dates[0])[:10]} -> {str(dates[n_train - 1])[:10]}; "
            f"kiem dinh: {str(dates[n_train])[:10]} -> {str(dates[-1])[:10]}")
    ket("khong co NaN trong r, X, RV",
        not (torch.isnan(d["r_all"]).any() or torch.isnan(d["X_all"]).any()
             or torch.isnan(d["rv_all"]).any()))
    ket("RV > 0 toan bo", bool((d["rv_all"] > 0).all()))
    return d


# ===========================================================================
# Phan 1: mo phong -- tinh lai tu checkpoint tung lan lap
# ===========================================================================
def _h_vo_dk(e: dict, s: int) -> float:
    return e[f"omega{s}"] / max(1.0 - e[f"alpha{s}"] - e[f"beta{s}"], 0.05)


def _doi_nhan(d: dict | None) -> dict | None:
    """Viet lai doc lap voi run_simulation._hoan_vi."""
    if d is None:
        return None
    ra = {}
    for k, v in d.items():
        m = re.fullmatch(r"(mu|omega|alpha|beta|nu)([12])", k)
        if m:
            ra[f"{m.group(1)}{3 - int(m.group(2))}"] = v
            continue
        m = re.fullmatch(r"gamma([12])_(\d+)", k)
        if m:
            ra[f"gamma{3 - int(m.group(1))}_{m.group(2)}"] = v
            continue
        if k == "p11":
            ra["p22"] = v
        elif k == "p22":
            ra["p11"] = v
        else:
            ra[k] = v
    return ra


def _sap_nhan(rep):
    e = rep.point_estimate
    if "omega1" not in e:
        return rep, False
    if _h_vo_dk(e, 1) <= _h_vo_dk(e, 2):
        return rep, False
    return type(rep)(rep.method, _doi_nhan(e), _doi_nhan(rep.ci_lower),
                     _doi_nhan(rep.ci_upper), rep.brier), True


def _that_ms(scn) -> dict:
    p = scn.params
    d = {}
    for ten in ("mu", "omega", "alpha", "beta"):
        v = getattr(p, ten)
        d[f"{ten}1"], d[f"{ten}2"] = float(v[0]), float(v[1])
    d["p11"], d["p22"] = float(scn.P0[0, 0]), float(scn.P0[1, 1])
    for s in range(2):
        for j in range(p.gamma.shape[1]):
            d[f"gamma{s + 1}_{j}"] = float(p.gamma[s, j])
    d["nu"] = float(p.nu)
    return d


def _tong_hop(reps: list, that: dict) -> dict:
    """Bias/RMSE/Coverage viet lai bang numpy thuan."""
    ra = {"bias": {}, "rmse": {}, "coverage": {}, "brier": None, "n": len(reps)}
    for ten, t0 in that.items():
        if ten not in reps[0].point_estimate:
            continue
        x = np.array([r.point_estimate[ten] for r in reps], dtype=float)
        ra["bias"][ten] = float((x - t0).mean())
        ra["rmse"][ten] = float(np.sqrt(((x - t0) ** 2).mean()))
        co = [r for r in reps if r.ci_lower is not None and ten in r.ci_lower]
        if co:
            lo = np.array([r.ci_lower[ten] for r in co], dtype=float)
            hi = np.array([r.ci_upper[ten] for r in co], dtype=float)
            ra["coverage"][ten] = float(((lo <= t0) & (t0 <= hi)).mean())
    br = [r.brier for r in reps if r.brier is not None]
    ra["brier"] = float(np.mean(br)) if br else None
    return ra


def _so_khoi(ten_kb: str, pp: str, tinh: dict, luu: dict, co_coverage: bool) -> None:
    """Doi chieu khoi (kich ban, phuong phap) da tinh lai voi khoi da luu."""
    if luu is None:
        ket(f"{ten_kb}/{pp}: co trong file tong hop", False, "thieu khoi")
        return
    n_luu = luu.get("n_replications")
    ket(f"{ten_kb}/{pp}: so lan lap {tinh['n']}", n_luu == tinh["n"], f"file ghi {n_luu}")
    for muc in ("bias", "rmse"):
        dmax, tham = 0.0, ""
        for ten, v in tinh[muc].items():
            if ten not in luu[muc]:
                ket(f"{ten_kb}/{pp}: {muc}[{ten}] co trong file", False)
                continue
            c = chenh(v, luu[muc][ten])
            if c > dmax:
                dmax, tham = c, ten
        ket(f"{ten_kb}/{pp}: {muc} ({len(tinh[muc])} tham so)", dmax <= TOL,
            f"lech lon nhat {dmax:.1e}" + (f" o {tham}" if tham else ""))
    if co_coverage:
        dmax = 0.0
        for ten, v in tinh["coverage"].items():
            if ten in luu.get("coverage", {}):
                dmax = max(dmax, chenh(v, luu["coverage"][ten]))
        thieu = [t for t in tinh["coverage"] if t not in luu.get("coverage", {})]
        ket(f"{ten_kb}/{pp}: coverage ({len(tinh['coverage'])} tham so)",
            dmax <= TOL and not thieu,
            f"lech lon nhat {dmax:.1e}" + (f"; thieu {thieu}" if thieu else ""))


def kiem_pr_a(spot: int) -> None:
    ghi("")
    ghi("=" * 78)
    ghi("PHAN 1A -- PR-A (MS-GARCH-X): tinh lai tu results/parts/sim/")
    ghi("=" * 78)
    ghi("  Quy uoc: sap nhan 'trang thai 1 = phuong sai vo dieu kien thap' (viet lai);")
    ghi("  gia tri that p11/p22 doc tu P0 cua kich ban, KHONG tu params.P (placeholder).")
    scns = default_scenarios()
    for kb, scn in scns.items():
        that = _that_ms(scn)
        h1, h2 = _h_vo_dk(that, 1), _h_vo_dk(that, 2)
        assert h1 < h2, f"{kb}: kich ban that khong theo quy uoc trang thai 1 thap"
        files = sorted(glob.glob(str(RES / "parts" / "sim" / f"{kb}_*.pt")))
        if not files:
            continue
        ghi(f"\n  --- {kb}: {scn.description[:60]}  (h1={h1:.3f}, h2={h2:.3f}, "
            f"p11={that['p11']}, p22={that['p22']}, nu={that['nu']})")
        tong = RES / f"07_simulation_{kb}.pt"
        luu = torch.load(tong, weights_only=False) if tong.exists() else None
        if luu is None:
            ghi(f"  (chua co {tong.name} -- chi tinh, khong doi chieu)")
        theo_pp: dict[str, list] = {}
        n_dao: dict[str, int] = {}
        for f in files:
            d = torch.load(f, weights_only=False)
            rep, dao = _sap_nhan(d["rep"])
            theo_pp.setdefault(d["method"], []).append(rep)
            n_dao[d["method"]] = n_dao.get(d["method"], 0) + int(dao)
        for pp, reps in theo_pp.items():
            tinh = _tong_hop(reps, that)
            if pp == "mcmc":
                ket(f"{kb}/mcmc: {tinh['n']} lan lap tham do, khong dua vao bang", True,
                    "chi co trong 07_simulation_thamdo/", canh_bao=True)
                continue
            ghi(f"  {pp}: {tinh['n']} lan lap, dao nhan {n_dao[pp]}, "
                f"brier(checkpoint goc, TRUOC sap nhan) = "
                f"{tinh['brier'] if tinh['brier'] is None else round(tinh['brier'], 4)}")
            if luu is not None:
                _so_khoi(kb, pp, tinh, luu.get(pp), co_coverage=(pp != "mle"))
                k = luu.get(pp, {})
                if "n_dao_nhan" in k:
                    ket(f"{kb}/{pp}: so lan dao nhan", k["n_dao_nhan"] == n_dao[pp],
                        f"tinh {n_dao[pp]}, file {k['n_dao_nhan']}")
                # Brier trong file tong hop: neu chua duoc thay bang ban bo sung
                # thi no la Brier TRUOC sap nhan -> sai, phai canh bao.
                if k.get("brier_da_sap_nhan"):
                    ket(f"{kb}/{pp}: Brier da tinh lai sau sap nhan", True,
                        f"{k['mean_brier']:.4f}")
                else:
                    ket(f"{kb}/{pp}: Brier trong file la ban TRUOC sap nhan (SAI)", False,
                        "chay bo_sung_coverage_brier.py --combine", canh_bao=False)
                if pp == "mle":
                    cov = k.get("coverage") or {}
                    ket(f"{kb}/mle: coverage Wald da co", bool(cov),
                        f"{len(cov)} tham so" if cov else "trong -- can bo_sung")
        _kiem_bo_sung(kb, theo_pp, that, scn, spot)


def _kiem_bo_sung(kb: str, theo_pp: dict, that: dict, scn, spot: int) -> None:
    """Doi chieu results/parts/bosung/ : coverage MLE & Brier da sap nhan."""
    bs_dir = RES / "parts" / "bosung"
    for pp in theo_pp:
        files = sorted(glob.glob(str(bs_dir / f"{kb}_{pp}_*.pt")))
        if not files:
            continue
        bs = [torch.load(f, weights_only=False) for f in files]
        briers = [x["brier"] for x in bs if "brier" in x]
        ghi(f"  bo_sung {kb}/{pp}: {len(bs)} file, Brier trung binh (sau sap nhan) = "
            f"{np.mean(briers):.4f}" if briers else f"  bo_sung {kb}/{pp}: {len(bs)} file")
        if briers:
            ket(f"{kb}/{pp}: Brier trong [0,2] (thang eq:brier-state)",
                all(0.0 <= b <= 2.0 for b in briers))
        if pp == "mle":
            ok = [x for x in bs if x.get("se_ok")]
            ket(f"{kb}/mle: SE Hessian tinh duoc", len(ok) == len(bs),
                f"{len(ok)}/{len(bs)}", canh_bao=(0 < len(ok) < len(bs)))
            if ok:
                cov = {}
                for ten, t0 in that.items():
                    lo = np.array([x["ci_lower"][ten] for x in ok if ten in x["ci_lower"]])
                    hi = np.array([x["ci_upper"][ten] for x in ok if ten in x["ci_upper"]])
                    if len(lo):
                        cov[ten] = float(((lo <= t0) & (t0 <= hi)).mean())
                ghi("    coverage MLE (tinh lai): " +
                    ", ".join(f"{k}={v:.2f}" for k, v in cov.items()))
                tong = RES / f"07_simulation_{kb}.pt"
                if tong.exists():
                    luu = torch.load(tong, weights_only=False).get("mle", {}).get("coverage") or {}
                    if luu:
                        dmax = max(chenh(cov[k], luu[k]) for k in cov if k in luu)
                        ket(f"{kb}/mle: coverage file tong hop = tinh lai", dmax <= TOL,
                            f"lech {dmax:.1e}")
                # CI Wald co the vuot bien (p_ss > 1, alpha < 0): chi la canh bao
                vuot = sum(1 for x in ok for k in ("p11", "p22")
                           if x["ci_upper"].get(k, 0) > 1.0)
                if vuot:
                    ket(f"{kb}/mle: CI Wald cua p_ss vuot 1 o {vuot} truong hop", True,
                        "dac diem cua khoang Wald, ghi chu trong bao cao", canh_bao=True)
    if spot > 0 and "mle" in theo_pp:
        _spot_brier(kb, scn, that, spot)
    if GRAD > 0 and "mle" in theo_pp:
        _kiem_gradient_mle(kb, scn, GRAD)


def _ms_params(e: dict, k: int) -> MSGARCHXParams:
    t = lambda *v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    return MSGARCHXParams(
        mu=t(e["mu1"], e["mu2"]), omega=t(e["omega1"], e["omega2"]),
        alpha=t(e["alpha1"], e["alpha2"]), beta=t(e["beta1"], e["beta2"]),
        gamma=torch.tensor([[e[f"gamma1_{j}"] for j in range(k)],
                            [e[f"gamma2_{j}"] for j in range(k)]], dtype=torch.float64),
        P=torch.tensor([[e["p11"], 1 - e["p11"]], [1 - e["p22"], e["p22"]]], dtype=torch.float64),
        nu=torch.tensor(e["nu"], dtype=torch.float64),
    )


def _spot_brier(kb: str, scn, that: dict, spot: int) -> None:
    """Tinh lai Brier TU DAU (sinh lai du lieu + bo loc) cho vai lan lap, doi
    chieu voi results/parts/bosung/. Ton ~40 s/lan lap voi MS-GARCH-X."""
    from msgarchx.simulation.dgp import simulate_ms_garch_x

    for pp in ("mle", "vb_mean_field", "vb_full_rank"):
        for b in range(spot):
            f_bs = RES / "parts" / "bosung" / f"{kb}_{pp}_{b:03d}.pt"
            f_sim = RES / "parts" / "sim" / f"{kb}_{pp}_{b:03d}.pt"
            if not (f_bs.exists() and f_sim.exists()):
                continue
            rep, _ = _sap_nhan(torch.load(f_sim, weights_only=False)["rep"])
            sim = simulate_ms_garch_x(scn.params, scn.P0, scn.T, n_exog=2,
                                      burn_in=scn.burn_in, seed=b)
            # S8: chuoi uoc luong la r + 0,3 r_{t-1} (viet lai, khong import)
            r = sim.r
            if scn.misspecified:
                r_tre = torch.roll(sim.r, 1)
                r_tre[0] = 0.0
                r = sim.r + 0.3 * r_tre
            with torch.no_grad():
                out = ms_garch_forward_filter(_ms_params(rep.point_estimate, 2), r, sim.X)
            xi = out["xi_filt"].numpy()
            ind = np.eye(2)[sim.state.numpy()]
            brier = float(((xi - ind) ** 2).sum(axis=1).mean())
            luu = torch.load(f_bs, weights_only=False)["brier"]
            ket(f"{kb}/{pp}/b={b}: Brier tinh lai tu dau = bo_sung", chenh(brier, luu) <= 1e-6,
                f"{brier:.4f} vs {luu:.4f}")


def _kiem_gradient_mle(kb: str, scn, so_lan: int) -> None:
    """|grad| cua -loglik tai nghiem MLE da luu: nghiem thuc su la diem dung
    thi gradient ~ 0. Gradient lon -> bo toi uu dung som; Hessian tai do khong
    phai Hessian tai cuc dai, nen SE Wald (neu tinh duoc) cung dang ngo."""
    from msgarchx.inference.mle import _neg_loglik
    from msgarchx.inference.param_spec import pack_params
    from msgarchx.simulation.dgp import simulate_ms_garch_x

    cfg = ModelConfig(n_exog=2, state_specific_nu=False)
    files = sorted(glob.glob(str(RES / "parts" / "sim" / f"{kb}_mle_*.pt")))[:so_lan]
    gnorm = []
    for f in files:
        d = torch.load(f, weights_only=False)
        rep, _ = _sap_nhan(d["rep"])
        b = d["b"]
        sim = simulate_ms_garch_x(scn.params, scn.P0, scn.T, n_exog=2, burn_in=scn.burn_in, seed=b)
        r = sim.r
        if scn.misspecified:
            r_tre = torch.roll(sim.r, 1); r_tre[0] = 0.0
            r = sim.r + 0.3 * r_tre
        raw = pack_params(_ms_params(rep.point_estimate, 2), cfg).clone().requires_grad_(True)
        loss, _ = _neg_loglik(raw, cfg, r, sim.X)
        g = torch.autograd.grad(loss, raw)[0]
        gnorm.append(float(g.abs().max()))
    gnorm = np.array(gnorm)
    ghi(f"  gradient |d(-loglik)/d raw|_max tai nghiem MLE ({len(gnorm)} lan lap): "
        f"trung vi {np.median(gnorm):.2e}, p90 {np.quantile(gnorm, 0.9):.2e}, max {gnorm.max():.2e}; "
        f"so lan > 1: {(gnorm > 1).sum()}, > 10: {(gnorm > 10).sum()}")
    ket(f"{kb}/mle: moi nghiem la diem dung (|grad|max < 1)", bool((gnorm < 1).all()),
        f"{(gnorm >= 1).sum()}/{len(gnorm)} lan lap co gradient >= 1 -> bo toi uu dung som",
        canh_bao=not bool((gnorm < 1).all()))


def kiem_pr_b() -> None:
    ghi("")
    ghi("=" * 78)
    ghi("PHAN 1B -- PR-B (EGARCH-X): tinh lai tu results/mo_phong_egarch/parts/")
    ghi("=" * 78)
    scns = default_egarch_scenarios()
    for kb, scn in scns.items():
        that = named_params_egarch(scn.params)
        files = sorted(glob.glob(str(RES / "mo_phong_egarch" / "parts" / f"{kb}_*.pt")))
        if not files:
            continue
        ghi(f"\n  --- {kb}: {scn.description[:60]}  that={ {k: round(v, 3) for k, v in that.items()} }")
        tong = RES / "mo_phong_egarch" / f"{kb}.pt"
        luu = torch.load(tong, weights_only=False) if tong.exists() else None
        theo_pp: dict[str, list] = {}
        for f in files:
            d = torch.load(f, weights_only=False)
            theo_pp.setdefault(d["method"], []).append(d["rep"])
        for pp, reps in theo_pp.items():
            tinh = _tong_hop(reps, that)
            ghi(f"  {pp}: {tinh['n']} lan lap")
            if luu is not None:
                _so_khoi(kb, pp, tinh, luu.get(pp), co_coverage=(pp != "mle"))
                k = luu.get(pp, {})
                if "tham_so_that" in k:
                    dmax = max(chenh(k["tham_so_that"][t], that[t]) for t in that)
                    ket(f"{kb}/{pp}: gia tri that trong file = kich ban", dmax <= TOL)
                if pp == "mle":
                    cov = k.get("coverage") or {}
                    ket(f"{kb}/mle: coverage Wald da co", bool(cov),
                        f"{len(cov)} tham so" if cov else "trong -- can bo_sung")
        # bo_sung cho PR-B: chi coverage MLE
        bs = [torch.load(f, weights_only=False)
              for f in sorted(glob.glob(str(RES / "parts" / "bosung" / f"{kb}_mle_*.pt")))]
        if bs:
            ok = [x for x in bs if x.get("se_ok")]
            ket(f"{kb}/mle: SE Hessian tinh duoc", len(ok) == len(bs), f"{len(ok)}/{len(bs)}",
                canh_bao=(0 < len(ok) < len(bs)))
            if ok:
                cov = {}
                for ten, t0 in that.items():
                    lo = np.array([x["ci_lower"][ten] for x in ok])
                    hi = np.array([x["ci_upper"][ten] for x in ok])
                    cov[ten] = float(((lo <= t0) & (t0 <= hi)).mean())
                ghi("    coverage MLE (tinh lai): " + ", ".join(f"{k}={v:.2f}" for k, v in cov.items()))
                if luu is not None and (luu.get("mle", {}).get("coverage") or {}):
                    lc = luu["mle"]["coverage"]
                    dmax = max(chenh(cov[k], lc[k]) for k in cov if k in lc)
                    ket(f"{kb}/mle: coverage file tong hop = tinh lai", dmax <= TOL, f"lech {dmax:.1e}")


# ===========================================================================
# Phan 2: Buoc 1 -- 8 mo hinh x 3 phuong phap tren du lieu thuc
# ===========================================================================
MA_CU = {"garch_n": "b1", "garch_t": "b2", "garch_x_t": "b3",
         "ms_garch_n": "b4", "ms_garch_x_t": "m1"}


def _nap_o(code: str, pp: str) -> dict | None:
    """Nap mot o (mo hinh, phuong phap) tu results/pipeline/ hoac tu ket qua cu."""
    p = RES / "pipeline" / f"{code}_{pp}.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    ma = MA_CU.get(code)
    if ma is None:
        return None
    if pp == "mle":
        m = torch.load(RES / "02_mle.pt", weights_only=False)
        o = torch.load(RES / "06_oos.pt", weights_only=False)
        if ma not in m:
            return None
        r = m[ma]
        return {"code": code, "method": "mle", "params": r.params, "loglik": r.loglik,
                "aic": r.aic, "bic": r.bic, "oos": o.get(ma, {}), "nap_lai": True,
                "config": getattr(r, "config", None)}
    if code == "ms_garch_x_t":
        v = torch.load(RES / "03_vb.pt", weights_only=False)
        o = torch.load(RES / "06_oos.pt", weights_only=False)
        kho = {"vb_mean_field": "m2", "vb_full_rank": "m3"}[pp]
        if kho not in v:
            return None
        return {"code": code, "method": pp, "final_elbo": v[kho].final_elbo,
                "samples": v[kho].samples, "oos": o.get(kho, {}), "nap_lai": True}
    return None


def _logf_don(params, family: str, dist: str, r: torch.Tensor, X: torch.Tensor, h0):
    fw = garch_forward if family == "garch" else egarch_forward
    h = fw(params, r, X, h0=h0)
    z = (r - params.mu) / torch.sqrt(h)
    if dist == "normal":
        return h, normal_logpdf(z) - 0.5 * torch.log(h)
    return h, std_t_logpdf(z, params.nu) - 0.5 * torch.log(h)


def _draws_don(samples: dict, spec, n: int) -> list:
    tong = samples["mu"].shape[0]
    buoc = max(1, tong // n)
    ra = []
    for i in range(0, tong, buoc):
        gamma = samples["gamma"][i] if "gamma" in samples else torch.zeros(0)
        nu = samples["nu_minus_2"][i] + 2.0 if spec.dist == "student_t" else None
        if spec.family == "garch":
            ra.append(GARCHParams(mu=samples["mu"][i], omega=samples["omega"][i],
                                  alpha=samples["alpha"][i], beta=samples["beta"][i],
                                  gamma=gamma, nu=nu))
        else:
            ra.append(EGARCHParams(mu=samples["mu"][i], omega=samples["omega"][i],
                                   alpha=samples["alpha"][i], theta=samples["theta"][i],
                                   beta=samples["beta"][i], gamma=gamma, nu=nu))
    return ra[:n]


def kiem_buoc_1(du_lieu: dict) -> None:
    ghi("")
    ghi("=" * 78)
    ghi("PHAN 2 -- BUOC 1: 8 mo hinh x 3 phuong phap tren VN30F1M")
    ghi("=" * 78)
    r_all, X_all, rv_all = du_lieu["r_all"], du_lieu["X_all"], du_lieu["rv_all"]
    n_train = int(du_lieu["n_train"])
    r_tr = r_all[:n_train]
    test = slice(n_train, None)
    rv_te, r_te = rv_all[test], r_all[test]
    N_te = int(rv_te.shape[0])
    bang = {(h["code"], h["phuong_phap"]): h
            for h in torch.load(RES / "pipeline" / "bang_so_sanh.pt", weights_only=False)}
    ket("bang_so_sanh.pt co 24 hang", len(bang) == 24, f"{len(bang)} hang")
    k = int(X_all.shape[1])

    def qlike(h):
        return float((torch.log(h) + rv_te / h).mean())

    for spec in model_spec.CATALOGUE:
        X = X_all if spec.use_exog else torch.zeros(X_all.shape[0], 0, dtype=X_all.dtype)
        X_tr = X[:n_train]
        for pp in ("mle", "vb_mean_field", "vb_full_rank"):
            o = _nap_o(spec.code, pp)
            ten = f"{spec.code}/{pp}"
            if o is None:
                ket(f"{ten}: co ket qua", False, "khong tim thay")
                continue
            oos = o.get("oos") or {}
            h_luu = oos.get("h_hat")
            hang = bang.get((spec.code, pp))
            ghi(f"\n  --- {ten}" + ("  (nap lai tu ket qua cu)" if o.get("nap_lai") else ""))

            # (a) bang so sanh = file o
            if hang is None:
                ket(f"{ten}: co trong bang_so_sanh", False)
            else:
                for cot, nguon in (("qlike", oos.get("qlike")), ("nls", oos.get("nls")),
                                   ("mae", oos.get("mae")), ("rmse", oos.get("rmse")),
                                   ("loglik", o.get("loglik")), ("aic", o.get("aic")),
                                   ("bic", o.get("bic")), ("elbo", o.get("final_elbo"))):
                    if nguon is None and hang.get(cot) is None:
                        continue
                    ket(f"{ten}: bang[{cot}] = file o", nguon is not None and hang.get(cot) is not None
                        and chenh(hang[cot], nguon) <= TOL)

            # (b) chi tieu ngoai mau tinh lai tu h_hat da luu (khong phu thuoc mo hinh)
            if h_luu is not None:
                h_luu = h_luu.double()
                ket(f"{ten}: n_test = {N_te}", int(h_luu.shape[0]) == N_te)
                ket(f"{ten}: QLIKE tu h_hat", chenh(qlike(h_luu), oos["qlike"]) <= TOL,
                    f"{qlike(h_luu):.6f}")
                ket(f"{ten}: MAE tu h_hat", chenh(float((rv_te - h_luu).abs().mean()), oos["mae"]) <= TOL)
                ket(f"{ten}: RMSE tu h_hat",
                    chenh(float(torch.sqrt(((rv_te - h_luu) ** 2).mean())), oos["rmse"]) <= TOL)

            # (c) tinh lai TU THAM SO: loglik (MLE), AIC/BIC, h_hat, NLS
            d_bang = hang["d"] if hang else spec.n_params(k)
            if pp == "mle":
                params = o["params"]
                if spec.n_regimes == 1:
                    with torch.no_grad():
                        h0 = torch.var(r_tr)
                        _, lf_tr = _logf_don(params, spec.family, spec.dist, r_tr, X_tr, h0)
                        ll = float(lf_tr[1:].sum())
                        h_all, lf_all = _logf_don(params, spec.family, spec.dist, r_all, X, h0)
                    ket(f"{ten}: loglik tinh lai tu tham so", chenh(ll, o["loglik"]) <= TOL_LOGLIK,
                        f"{ll:.6f} vs {o['loglik']:.6f}")
                    nls = float(-lf_all[test].mean())
                    ket(f"{ten}: NLS tinh lai tu tham so", chenh(nls, oos["nls"]) <= TOL_LOGLIK,
                        f"{nls:.6f} vs {oos['nls']:.6f}")
                    if h_luu is not None:
                        ket(f"{ten}: h_hat tinh lai = h_hat da luu",
                            float((h_all[test] - h_luu).abs().max()) <= 1e-8)
                else:
                    cfg = o.get("config") or ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
                    with torch.no_grad():
                        out = ms_garch_forward_filter(params, r_tr, X_tr)
                    ll = float(out["loglik"])
                    ket(f"{ten}: loglik tinh lai tu tham so (bo loc tien)",
                        chenh(ll, o["loglik"]) <= 1e-4, f"{ll:.6f} vs {o['loglik']:.6f}")
                ll_luu = float(o["loglik"])
                ket(f"{ten}: AIC = 2d - 2 loglik (d={d_bang})",
                    chenh(2 * d_bang - 2 * ll_luu, o["aic"]) <= 1e-6,
                    f"{2 * d_bang - 2 * ll_luu:.4f} vs {o['aic']:.4f}")
                bic = d_bang * math.log(n_train) - 2 * ll_luu
                ket(f"{ten}: BIC = d ln(n_train) - 2 loglik", chenh(bic, o["bic"]) <= 1e-6,
                    f"{bic:.4f} vs {o['bic']:.4f}")
            else:
                if spec.n_regimes == 1 and "samples" in o:
                    draws = _draws_don(o["samples"], spec, 200)
                    with torch.no_grad():
                        h0 = torch.var(r_tr)
                        hs, lfs = [], []
                        for p in draws:
                            h_, lf_ = _logf_don(p, spec.family, spec.dist, r_all, X, h0)
                            hs.append(h_[test]); lfs.append(lf_[test])
                        H, LF = torch.stack(hs), torch.stack(lfs)
                    h_mean = H.mean(0)
                    nls = float(-(torch.logsumexp(LF, 0) - math.log(LF.shape[0])).mean())
                    ket(f"{ten}: h_hat (trung binh {len(draws)} mau) tinh lai = da luu",
                        h_luu is not None and float((h_mean - h_luu).abs().max()) <= 1e-8)
                    ket(f"{ten}: NLS hon hop tinh lai tu mau hau nghiem",
                        chenh(nls, oos["nls"]) <= TOL_LOGLIK, f"{nls:.6f} vs {oos['nls']:.6f}")
                else:
                    ket(f"{ten}: NLS/h_hat (mo hinh 2 trang thai, VB)", True,
                        "chi doi chieu tu h_hat da luu; khong tinh lai tu mau", canh_bao=True)
            # (d) VaR: so vi pham khop voi kiem dinh Kupiec da luu
            for lab, alpha in (("1pct", 0.01), ("5pct", 0.05)):
                kp = oos.get(f"kupiec_{lab}")
                if kp:
                    ket(f"{ten}: Kupiec {lab}: n_vi_pham/n_obs nhat quan",
                        kp["n_obs"] == N_te and kp["n_violations"] == oos.get(f"n_violations_{lab}"),
                        f"{kp['n_violations']}/{kp['n_obs']} (ky vong {alpha * N_te:.1f}), p={kp['p_value']:.3f}")

    # (e) mo hinh thang cuoc theo tung tieu chi
    ghi("")
    for tc in ("aic", "bic", "qlike", "nls"):
        hop_le = [h for h in bang.values() if isinstance(h.get(tc), (int, float))]
        best = min(hop_le, key=lambda h: h[tc])
        ghi(f"  tot nhat theo {tc.upper():5}: {best['code']}/{best['phuong_phap']} = {best[tc]:.4f}")
    b = bang[("egarch_x_t", "mle")]
    ket("EGARCH-X-t/MLE thang ca AIC, BIC, QLIKE, NLS",
        all(b[tc] == min(h[tc] for h in bang.values() if isinstance(h.get(tc), (int, float)))
            for tc in ("aic", "bic", "qlike", "nls")))


# ===========================================================================
# Phan 3: Buoc 3 -- MCMC tren mo hinh thang cuoc
# ===========================================================================
def _split_rhat(x: np.ndarray) -> float:
    """R-hat tach doi (Gelman et al. 2013), viet lai; x: (chains, draws)."""
    c, n = x.shape
    h = n // 2
    x = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
    m, n = x.shape
    mean_c = x.mean(axis=1)
    B = n * mean_c.var(ddof=1)
    W = x.var(axis=1, ddof=1).mean()
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def kiem_buoc_3() -> None:
    ghi("")
    ghi("=" * 78)
    ghi("PHAN 3 -- BUOC 3: MCMC (NUTS) tren EGARCH-X Student-t")
    ghi("=" * 78)
    p = RES / "pipeline" / "mcmc_egarch_x_t.pt"
    if not p.exists():
        ket("co mcmc_egarch_x_t.pt", False)
        return
    m = torch.load(p, weights_only=False)
    S, sm = m["samples"], m["summary"]
    n = int(S["mu"].shape[0])
    ghi(f"  {n} mau, max_rhat={m['max_rhat']:.4f}, min_ess={m['min_ess']:.1f}, "
        f"phan ky={m['n_divergences']}, {m['giay'] / 3600:.1f} gio")
    ket("so mau = 4 chuoi x 400", n == 1600, f"{n}")
    ket("khong phan ky", m["n_divergences"] == 0)
    ket("max R-hat < 1.05", m["max_rhat"] < 1.05, f"{m['max_rhat']:.4f}")
    ket("min ESS > 400", m["min_ess"] > 400, f"{m['min_ess']:.0f}")
    dmax = 0.0
    for ten, s in sm.items():
        x = S[ten].double()
        if x.dim() == 1:
            x = x[:, None]
        for j in range(x.shape[1]):
            col = x[:, j].numpy()
            mean, sd = col.mean(), col.std(ddof=1)
            q = np.quantile(col, [0.025, 0.975])
            luu = {k: (float(v[j]) if torch.as_tensor(v).dim() else float(v)) for k, v in s.items()}
            dmax = max(dmax, chenh(mean, luu["mean"]), chenh(q[0], luu["q2.5"]), chenh(q[1], luu["q97.5"]))
            # sd cua torch.std mac dinh ddof=1; pyro co the dung ddof=0 -> nhan ca hai
            d_sd = min(chenh(sd, luu["sd"]), chenh(col.std(ddof=0), luu["sd"]))
            dmax = max(dmax, d_sd)
    ket("summary (mean, sd, q2.5, q97.5) = tinh lai tu mau", dmax <= 1e-3, f"lech lon nhat {dmax:.1e}")
    # R-hat tinh lai: gia dinh mau xep theo chuoi (chain-major) nhu pyro tra ve
    rh = {}
    for ten in S:
        x = S[ten].double()
        if x.dim() == 1:
            x = x[:, None]
        for j in range(x.shape[1]):
            rh[f"{ten}{'' if x.shape[1] == 1 else j}"] = _split_rhat(x[:, j].numpy().reshape(4, 400))
    ghi("  R-hat tach doi tinh lai: " + ", ".join(f"{k}={v:.4f}" for k, v in rh.items()))
    ket("max R-hat tinh lai < 1.05 (gia dinh mau xep theo chuoi)", max(rh.values()) < 1.05,
        f"{max(rh.values()):.4f} (file: {m['max_rhat']:.4f})", canh_bao=abs(max(rh.values()) - m["max_rhat"]) > 0.01)
    th = sm["theta"]
    ket("theta < 0 voi CI 95% khong chua 0", float(th["q97.5"]) < 0,
        f"[{float(th['q2.5']):.4f}, {float(th['q97.5']):.4f}]")
    # R_sd so voi VB
    for pp in ("vb_mean_field", "vb_full_rank"):
        f = RES / "pipeline" / f"egarch_x_t_{pp}.pt"
        if f.exists():
            v = torch.load(f, weights_only=False)
            rs = {}
            for ten in ("omega", "alpha", "theta", "beta"):
                sd_vb = float(v["samples"][ten].double().std())
                sd_mc = float(S[ten].double().std())
                rs[ten] = sd_vb / sd_mc
            ghi(f"  R_sd = sd_{pp}/sd_MCMC: " + ", ".join(f"{k}={x:.2f}" for k, x in rs.items()))


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phan", choices=["sim", "buoc1", "buoc3", "all"], default="all")
    ap.add_argument("--spot", type=int, default=0,
                    help="so lan lap moi kich ban tinh lai Brier tu dau (cham)")
    ap.add_argument("--grad", type=int, default=0,
                    help="so lan lap moi kich ban kiem tra gradient tai nghiem MLE (~1 s/lan)")
    a = ap.parse_args()
    global GRAD
    GRAD = a.grad

    ghi(f"KIEM TOAN KET QUA -- {Path(__file__).name}")
    du_lieu = kiem_du_lieu()
    if a.phan in ("sim", "all"):
        kiem_pr_a(a.spot)
        kiem_pr_b()
    if a.phan in ("buoc1", "all"):
        kiem_buoc_1(du_lieu)
    if a.phan in ("buoc3", "all"):
        kiem_buoc_3()

    ghi("")
    ghi("=" * 78)
    ghi(f"TONG KET: {TONG['dat']} dat, {TONG['hong']} hong, {TONG['canh_bao']} canh bao")
    ghi("=" * 78)
    out = RES / "kiem_toan_ket_qua.txt"
    out.write_text("\n".join(DONG) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
