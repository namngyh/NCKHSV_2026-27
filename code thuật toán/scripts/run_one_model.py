"""Chay MOT mo hinh rieng le trong ke hoach so sanh cac mo hinh MLE (so tay
danh so 01-20; xem cai_dat_va_chay_full.bat / run_full_study.py cho quy trinh
day du chay CA GOI mot luot).

Muc tieu cua script nay KHAC voi run_full_study.py:
  - Ban tu chon mo hinh nao muon chay va luc nao chay, khong bat buoc theo
    thu tu 1->20 va khong phai cho chay het toan bo goi.
  - Tien do in TRUC TIEP ra terminal qua verbose=True (loglik moi 100-200
    vong lap, giong het cach run_full_study.py da lam).
  - Ket qua duoc GHI THEM (append) vao MOT file .txt duy nhat
    (results/screening_log.txt). File nay KHONG BAO GIO bi xoa hay ghi de --
    moi lan chay (du la tinh moi hay nap lai tu checkpoint san co) deu them
    mot khoi moi kem dau thoi gian, nen lich su cac lan chay truoc van con.
  - Ket qua so (GARCHMLEResult / MLEResult) con duoc luu rieng dang .pt
    trong results/screening/ de tai su dung sau nay (vd. chon "mo hinh thang
    cuoc" roi dua vao Variational Bayes) ma khong phai tinh lai.

Nhom mo hinh GARCH-X / MS-GARCH-X (05-07, 12-14) hien chi dung 2 bien ngoai
sinh da co san trong pipeline du lieu (log_volume_lag1, log_rv_lag1) -> 3 to
hop khac rong. Cac muc 08-11 va 15-18 (bien thu 3 tro len: basis, ngay dao
han, ngay chuyen hop dong) DA BO theo yeu cau, vi du lieu tho tuong ung
chua co trong ohlc_export.csv. Muc 19 (HAR-RV) va 20 (Realized GARCH) cung
chua chay duoc vi chua co model tuong ung trong msgarchx/models/.

Chay: python scripts/run_one_model.py --list
      python scripts/run_one_model.py 1
      python scripts/run_one_model.py 5 --n-starts 2 --n-adam 150 --n-lbfgs 80
      python scripts/run_one_model.py 3 --force
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from msgarchx.data.exogenous import build_exogenous_matrix, standardize_with_train_stats
from msgarchx.data.loader import load_vn30f1m_daily
from msgarchx.inference.garch_mle import GARCHMLEResult, fit_garch_mle
from msgarchx.inference.mle import MLEResult, fit_mle
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.models.ms_garch import label_switch_correction

DEFAULT_RESULTS_DIR = ROOT / "results"
SCREEN_SUBDIR = "screening"
LOG_FILE_NAME = "screening_log.txt"

# Thu tu cot CO DINH trong X_all cua 00_data.pt (khop prepare_data trong
# run_full_study.py). Neu sau nay them bien ngoai sinh moi, cap nhat o day.
EXOG_COLS = ["log_volume_lag1", "log_rv_lag1"]


# ============================================================
# Dac ta 20 mo hinh (so tay)
# ============================================================
@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    ready: bool
    note: str = ""
    kind: str = ""                    # "garch" | "ms_garch"
    dist: str = ""                    # "normal" | "student_t" (garch)
    fixed_nu: float | None = None     # ms_garch: None = nu tu do
    exog_cols: tuple[str, ...] = ()


def _r(id_, label, kind, **kw) -> ModelSpec:
    return ModelSpec(id=id_, label=label, ready=True, kind=kind, **kw)


def _pending(id_, label, note) -> ModelSpec:
    return ModelSpec(id=id_, label=label, ready=False, note=note)


REGISTRY: dict[str, ModelSpec] = {
    s.id: s for s in [
        _r("01", "GARCH(1,1)-n (1 trang thai, chuan, khong X)", "garch", dist="normal"),
        _r("02", "GARCH(1,1)-t (1 trang thai, Student-t, khong X)", "garch", dist="student_t"),
        _r("03", "MS-GARCH(1,1)-n (2 trang thai, nu co dinh ~ chuan, khong X)", "ms_garch", fixed_nu=1e6),
        _r("04", "MS-GARCH(1,1)-t (2 trang thai, Student-t tu do, khong X)", "ms_garch", fixed_nu=None),
        _r("05", "GARCH-X-n (1 trang thai, chuan, X = log_volume_lag1)", "garch", dist="normal",
           exog_cols=("log_volume_lag1",)),
        _r("06", "GARCH-X-n (1 trang thai, chuan, X = log_rv_lag1)", "garch", dist="normal",
           exog_cols=("log_rv_lag1",)),
        _r("07", "GARCH-X-n (1 trang thai, chuan, X = log_volume_lag1 + log_rv_lag1)", "garch", dist="normal",
           exog_cols=("log_volume_lag1", "log_rv_lag1")),
        _pending("08-11", "GARCH-X-n (bien thu 3 tro len: basis/expiry/roll)",
                 "Da bo theo yeu cau -- du lieu tho tuong ung chua co trong ohlc_export.csv."),
        _r("12", "MS-GARCH-X-n (2 trang thai, nu co dinh, X = log_volume_lag1)", "ms_garch", fixed_nu=1e6,
           exog_cols=("log_volume_lag1",)),
        _r("13", "MS-GARCH-X-n (2 trang thai, nu co dinh, X = log_rv_lag1)", "ms_garch", fixed_nu=1e6,
           exog_cols=("log_rv_lag1",)),
        _r("14", "MS-GARCH-X-n (2 trang thai, nu co dinh, X = log_volume_lag1 + log_rv_lag1)", "ms_garch",
           fixed_nu=1e6, exog_cols=("log_volume_lag1", "log_rv_lag1")),
        _pending("15-18", "MS-GARCH-X-n (bien thu 3 tro len)",
                 "Da bo theo yeu cau -- cung ly do voi 08-11."),
        _pending("19", "HAR-RV", "Chua co model tuong ung trong msgarchx/models/ -- can viet moi truoc."),
        _pending("20", "Realized GARCH", "Chua co model tuong ung trong msgarchx/models/ -- can viet moi truoc."),
    ]
}


def _print_registry() -> None:
    print("Danh sach 20 mo hinh (so tay ke hoach so sanh):\n")
    for key in sorted(REGISTRY, key=lambda k: int(k.split("-")[0])):
        spec = REGISTRY[key]
        status = "SAN SANG" if spec.ready else "khong chay duoc"
        print(f"  #{key:<6} [{status:>16}]  {spec.label}")
        if not spec.ready:
            print(f"          -> {spec.note}")
    print()
    print("Muc 21 (mo hinh thang cuoc -> Variational Bayes) va 22 (-> MCMC) khong")
    print("nam trong script nay: chung chay TREN mo hinh da chon sau khi so sanh")
    print("ket qua 01-14, khong phai mot dac ta co dinh tu dau.")
    print()
    print("Vi du: python scripts/run_one_model.py 5")


# ============================================================
# Du lieu (dung chung 00_data.pt voi run_full_study.py, tu tinh neu chua co)
# ============================================================
def load_data(results_dir: Path) -> dict:
    path = results_dir / "00_data.pt"
    if path.exists():
        print(f"[du lieu] Nap tu checkpoint co san: {path}")
        return torch.load(path, weights_only=False)

    print("[du lieu] Chua co checkpoint 00_data.pt, dang doc ohlc_export.csv ...")
    daily, quality_report = load_vn30f1m_daily(str(ROOT / "ohlc_export.csv"))
    X_raw = build_exogenous_matrix(
        daily, volume_col="volume", rv_col="RV",
        basis_col=None, expiry_dummy_col=None, roll_dummy_col=None,
    )
    n = len(daily)
    train_mask = pd.Series(False, index=X_raw.index)
    train_mask.iloc[: int(n * 0.8)] = True
    X_scaled, x_mean, x_std = standardize_with_train_stats(X_raw, train_mask, exclude_cols=("date",))

    valid = (
        daily["log_return"].notna()
        & X_scaled["log_volume_lag1"].notna()
        & X_scaled["log_rv_lag1"].notna()
        & daily["RV"].notna()
    )
    r_all = torch.tensor(daily.loc[valid, "log_return"].to_numpy())
    X_all = torch.tensor(X_scaled.loc[valid, EXOG_COLS].to_numpy())
    rv_all = torch.tensor(daily.loc[valid, "RV"].to_numpy())
    dates_all = daily.loc[valid, "date"].reset_index(drop=True)
    n_train = int(train_mask.loc[valid].sum())

    data = {
        "r_all": r_all, "X_all": X_all, "rv_all": rv_all, "dates_all": dates_all,
        "n_train": n_train, "quality_report": quality_report,
        "exog_mean": x_mean, "exog_std": x_std,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"[du lieu] Da luu checkpoint moi: {path}")
    return data


# ============================================================
# Fit
# ============================================================
def _select_exog(X_train_full: torch.Tensor, cols: tuple[str, ...]) -> torch.Tensor:
    if not cols:
        return torch.zeros(X_train_full.shape[0], 0, dtype=X_train_full.dtype)
    idx = [EXOG_COLS.index(c) for c in cols]
    return X_train_full[:, idx]


def fit_one(spec: ModelSpec, data: dict, n_starts: int, n_adam: int, n_lbfgs: int,
            base_seed: int, verbose: bool):
    r_all, X_all, n_train = data["r_all"], data["X_all"], data["n_train"]
    r_train = r_all[:n_train]
    X_train = _select_exog(X_all[:n_train], spec.exog_cols)

    if spec.kind == "garch":
        return fit_garch_mle(r_train, X_train, dist=spec.dist, n_starts=n_starts,
                              base_seed=base_seed, n_adam=n_adam, n_lbfgs=n_lbfgs, verbose=verbose)
    if spec.kind == "ms_garch":
        cfg = ModelConfig(n_exog=X_train.shape[1], state_specific_nu=False)
        return fit_mle(r_train, X_train, cfg, n_starts=n_starts, base_seed=base_seed,
                        n_adam=n_adam, n_lbfgs=n_lbfgs, verbose=verbose, fixed_nu=spec.fixed_nu)
    raise ValueError(f"kind khong hop le: {spec.kind!r}")


# ============================================================
# Dinh dang khoi ket qua ghi vao .txt
# ============================================================
def format_block(spec: ModelSpec, res, elapsed: float, from_cache: bool, cfg_desc: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = "(nap tu checkpoint co san)" if from_cache else "(vua tinh moi)"
    lines = [
        "=" * 78,
        f"[{ts}] Mo hinh #{spec.id} -- {spec.label} {tag}",
        "=" * 78,
        cfg_desc,
        f"Thoi gian chay (lan tinh goc): {elapsed/60:.1f} phut ({elapsed:.0f}s)",
        "",
    ]

    if isinstance(res, GARCHMLEResult):
        p = res.params
        nu_s = f"{p.nu.item():.4f}" if p.nu is not None else "-- (phan phoi chuan)"
        lines.append(f"Log-hop-ly = {res.loglik:.2f}   AIC = {res.aic:.2f}   BIC = {res.bic:.2f}")
        lines.append(f"mu={p.mu.item():.4f}  omega={p.omega.item():.4f}  alpha={p.alpha.item():.4f}  "
                     f"beta={p.beta.item():.4f}  nu={nu_s}")
        if p.gamma.numel():
            gamma_s = ", ".join(f"{c}={v:.4f}" for c, v in zip(spec.exog_cols, p.gamma.tolist()))
            lines.append(f"gamma: {gamma_s}")
        lines.append("")
        lines.append("Chan doan cac lan khoi tao:")
        for r in res.runs:
            lines.append(f"  seed={r.seed}  loglik={r.loglik:.2f}  hoi_tu={'co' if r.converged else 'KHONG'}  "
                        f"||grad||={r.grad_norm:.4f}")
    elif isinstance(res, MLEResult):
        p = res.params
        h = res.filter_output["h"] if res.filter_output is not None else None
        p_c = label_switch_correction(p, h) if h is not None else p
        lines.append(f"Log-hop-ly = {res.loglik:.2f}   AIC = {res.aic:.2f}   BIC = {res.bic:.2f}")
        lines.append(f"mu    = {[round(v, 4) for v in p_c.mu.tolist()]}")
        lines.append(f"omega = {[round(v, 4) for v in p_c.omega.tolist()]}")
        lines.append(f"alpha = {[round(v, 4) for v in p_c.alpha.tolist()]}")
        lines.append(f"beta  = {[round(v, 4) for v in p_c.beta.tolist()]}")
        if p_c.gamma.numel():
            lines.append(f"gamma = {[round(v, 4) for v in p_c.gamma.reshape(-1).tolist()]}  (X = {list(spec.exog_cols)})")
        lines.append(f"P     = {[[round(v, 4) for v in row] for row in p_c.P.tolist()]}")
        nu_s = "co dinh ~ chuan" if spec.fixed_nu is not None else f"{p_c.nu.item():.4f}"
        lines.append(f"nu    = {nu_s}")
        lines.append("")
        lines.append("Chan doan cac lan khoi tao:")
        for r in res.runs:
            lines.append(f"  seed={r.seed}  loglik={r.loglik:.2f}  hoi_tu={'co' if r.converged else 'KHONG'}  "
                        f"||grad||={r.grad_norm:.4f}  gan_bien={'co' if r.near_boundary else 'khong'}")
    else:
        lines.append(f"(khong nhan dien duoc kieu ket qua: {type(res)})")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", nargs="?", help="So mo hinh (01-20), vd: 5 hoac 05")
    parser.add_argument("--list", action="store_true", help="Liet ke toan bo mo hinh va trang thai san sang")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--n-starts", type=int, default=6)
    parser.add_argument("--n-adam", type=int, default=500)
    parser.add_argument("--n-lbfgs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Chay lai du checkpoint da co san")
    parser.add_argument("--quiet", action="store_true", help="Tat in tien do chi tiet tung vong lap")
    args = parser.parse_args()

    if args.list or not args.model_id:
        _print_registry()
        return 0

    key = args.model_id.zfill(2)
    if key not in REGISTRY:
        print(f"[LOI] Khong co mo hinh #{key}. Dung --list de xem danh sach.", file=sys.stderr)
        return 1
    spec = REGISTRY[key]
    if not spec.ready:
        print(f"[CHUA CHAY DUOC] Mo hinh #{key} -- {spec.label}", file=sys.stderr)
        print(f"Ly do: {spec.note}", file=sys.stderr)
        return 1

    results_dir = Path(args.results_dir)
    screen_dir = results_dir / SCREEN_SUBDIR
    screen_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / LOG_FILE_NAME
    ckpt_path = screen_dir / f"{key}.pt"

    data = load_data(results_dir)

    from_cache = ckpt_path.exists() and not args.force
    if from_cache:
        print(f"[bo qua tinh toan] Da co checkpoint: {ckpt_path} (dung --force de chay lai)")
        payload = torch.load(ckpt_path, weights_only=False)
        res, elapsed, cfg_desc = payload["result"], payload["elapsed"], payload["cfg_desc"]
        print(f"[luu y] Checkpoint nay duoc tinh voi: {cfg_desc}")
    else:
        cfg_desc = (f"Cau hinh MLE: n_starts={args.n_starts}, n_adam={args.n_adam}, "
                    f"n_lbfgs={args.n_lbfgs}, seed={args.seed}")
        print(f"[chay] Mo hinh #{key} -- {spec.label}")
        print(cfg_desc)
        t0 = time.time()
        res = fit_one(spec, data, args.n_starts, args.n_adam, args.n_lbfgs, args.seed, verbose=not args.quiet)
        elapsed = time.time() - t0
        torch.save({"result": res, "elapsed": elapsed, "cfg_desc": cfg_desc}, ckpt_path)
        print(f"[xong] {elapsed/60:.1f} phut -> {ckpt_path}")

    block = format_block(spec, res, elapsed, from_cache, cfg_desc)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"\n[ghi log] Da THEM ket qua vao: {log_path} (khong xoa noi dung cu)\n")
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
