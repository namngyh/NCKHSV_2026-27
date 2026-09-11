"""Kiem tra: 800 buoc SVI trong mo phong (PR-A) da DU de VB hoi tu chua?

VI SAO CAN KIEM TRA. Nghien cuu mo phong dung `sim_vb_n_steps = 800`, trong khi
nghien cuu chinh tren du lieu that dung 3000. Neu 800 buoc CHUA hoi tu thi con
so "do chech cua VB-MF" ma PR-A bao cao dang tron hai thu khac han nhau:
sai so xap xi mean-field THAT, va don gian la toi uu hoa chua xong.

Dieu nay de doa truc tiep phat hien chu luc "coverage ~ 0,33": guide AutoNormal
khoi tao tham so thang do (scale) o gia tri NHO roi no dan trong qua trinh hoc.
Mot guide dung som se cho khoang tin cay QUA HEP -- dung trieu chung dang duoc
dien giai la "mean-field danh gia thap bat dinh". Neu scale van con dang no o
buoc 800 thi cach dien giai do chua vung.

CACH DO. Chay MOT lan lap den 3000 buoc, doc thang tham so scale cua guide tai
cac moc. Doc scale KHONG ton gi (chi la doc tensor trong param store), khac han
viec lay mau hau nghiem bang Predictive -- moi mau phai chay lai bo loc tien
qua 2000 quan sat, 500 mau la them ~13 phut.

Chay:
    python scripts/kiem_tra_hoi_tu_vb.py
    python scripts/kiem_tra_hoi_tu_vb.py --n-steps 3000 --kich-ban S1 --b 0
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

import pyro  # noqa: E402
from pyro.infer import SVI, Trace_ELBO  # noqa: E402
from pyro.infer.autoguide import AutoNormal  # noqa: E402

from msgarchx.inference.bayes_model import model as bayes_model  # noqa: E402
from msgarchx.inference.param_spec import ModelConfig, PriorConfig  # noqa: E402
from msgarchx.simulation.dgp import simulate_ms_garch_x  # noqa: E402
from msgarchx.simulation.scenarios import default_scenarios  # noqa: E402

# Cac moc de doi chieu. 800 la gia tri PR-A dang dung; 3000 la gia tri nghien
# cuu chinh dung tren du lieu that.
MOC = [100, 200, 400, 800, 1500, 2000, 3000]


def _scales() -> dict[str, float]:
    """Do lech chuan hau nghiem cua guide, tren thang KHONG rang buoc.

    AutoNormal luu tham so voi ten "AutoNormal.scales.<ten_site>". Voi site
    nhieu chieu ta lay trung binh -- du de tra loi cau hoi "con dang no khong",
    khong can chi tiet tung toa do.
    """
    ra = {}
    for ten, gt in pyro.get_param_store().items():
        if ".scales." in ten:
            ra[ten.split(".scales.")[-1]] = float(gt.detach().mean())
    return ra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--kich-ban", dest="kich_ban", default="S1")
    ap.add_argument("--b", type=int, default=0, help="chi so lan lap (dung lam seed)")
    ap.add_argument("--T", type=int, default=2000)
    args = ap.parse_args()

    scenario = default_scenarios(T_default=args.T, B_default=1)[args.kich_ban]
    config = ModelConfig(n_exog=2, state_specific_nu=False)
    prior = PriorConfig()

    sim = simulate_ms_garch_x(scenario.params, scenario.P0, scenario.T,
                              n_exog=2, burn_in=scenario.burn_in, seed=args.b)

    print(f"Kiem tra hoi tu VB mean-field | {args.kich_ban}, seed={args.b}, "
          f"T={scenario.T}, den {args.n_steps} buoc")
    print("(PR-A dang dung 800 buoc; nghien cuu chinh dung 3000)\n")

    pyro.set_rng_seed(args.b)
    pyro.clear_param_store()
    guide = AutoNormal(bayes_model)
    svi = SVI(bayes_model, guide, pyro.optim.Adam({"lr": 0.01}),
              loss=Trace_ELBO(num_particles=1))

    lich_su: dict[int, tuple[float, dict[str, float]]] = {}
    t0 = time.time()
    for buoc in range(1, args.n_steps + 1):
        loss = svi.step(sim.r, sim.X, config, prior, "clamp", 1e-8)
        if buoc in MOC:
            lich_su[buoc] = (-loss, _scales())
            print(f"  buoc {buoc:>5}  ELBO={-loss:>12.2f}  "
                  f"({(time.time()-t0)/60:.1f} phut)", flush=True)

    ten_site = sorted(next(iter(lich_su.values()))[1])
    print(f"\n{'THANG DO HAU NGHIEM (scale) THEO BUOC':^76}")
    print(f"  {'site':<14}" + "".join(f"{m:>9}" for m in MOC if m in lich_su))
    for site in ten_site:
        hang = "".join(f"{lich_su[m][1][site]:>9.4f}" for m in MOC if m in lich_su)
        print(f"  {site:<14}{hang}")

    if 800 in lich_su and args.n_steps >= 3000 and 3000 in lich_su:
        print("\n" + "=" * 76)
        print("  KET LUAN: ty le scale(3000) / scale(800) theo tung site")
        print("=" * 76)
        ty_le = {s: lich_su[3000][1][s] / lich_su[800][1][s] for s in ten_site}
        for s in sorted(ty_le, key=lambda x: -ty_le[x]):
            canh_bao = "  <-- con no manh" if ty_le[s] > 1.25 else ""
            print(f"  {s:<14}{ty_le[s]:>8.2f}x{canh_bao}")
        lon_nhat, nho_nhat = max(ty_le.values()), min(ty_le.values())
        print()
        # Ba ket cuc, khong phai hai. Ban dau ham nay chi kiem tra "co site nao
        # no qua 25% khong" -- va da doc SAI chinh du lieu no sinh ra: khi cac
        # ty le di CA HAI CHIEU (co site no ra, co site co lai) thi do la dau
        # hieu NHIEU, khong phai dau hieu dung som. Nang so buoc khong chua duoc
        # nhieu; phai giam phuong sai gradient (tang num_particles) hoac ha dan
        # buoc hoc.
        if lon_nhat > 1.25 and nho_nhat < 0.80:
            print("  => KHONG KET LUAN duoc theo huong nao. Ty le di ca hai chieu")
            print(f"     (cao nhat {lon_nhat:.2f}x, thap nhat {nho_nhat:.2f}x), tuc thang do")
            print("     KHONG on dinh o CA 800 LAN 3000 buoc -- toi uu hoa con dang di")
            print("     lang quang chu chua dung lai. Nang so buoc se khong giai quyet;")
            print("     xem lai num_particles va lich buoc hoc thi dung huong hon.")
        elif lon_nhat > 1.25:
            print("  => 800 buoc CHUA DU. Moi ty le deu >= 0,80 va co site no manh, tuc")
            print("     khoang tin cay o buoc 800 HEP HON mot cach he thong. Coverage")
            print("     thap cua PR-A mot phan la do DUNG SOM. Nen nang sim_vb_n_steps.")
        else:
            print("  => 800 buoc DA DU (moi scale on dinh trong pham vi 25%).")
            print("     Coverage thap khong phai do dung som -- ket luan ve mean-field vung.")
        d_elbo = lich_su[3000][0] - lich_su[800][0]
        print(f"\n  ELBO tang them tu buoc 800 den 3000: {d_elbo:+.2f}")


if __name__ == "__main__":
    main()
