"""Bang theo doi tien trinh TRUC TIEP cua moi thu dang chay.

Doc tu cac tep nhat ky trong `logs/` va cac checkpoint trong `results/`, khong
can gan vao tien trinh nao nen chay duoc bat cu luc nao, bao nhieu cua so cung
duoc, va khong lam anh huong gi den cac tien trinh dang chay.

Chay:
    python scripts/tien_do.py            # in mot lan roi thoat
    python scripts/tien_do.py --lap      # cap nhat lien tuc moi 30 giay
    python scripts/tien_do.py --lap --moi 10
Hoac double-click `xem_tien_do.bat`.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

_TQDM = re.compile(
    r"(?P<pha>Warmup|Sample):\s+(?P<pct>\d+)%.*?\|\s*(?P<i>\d+)/(?P<n>\d+)\s*\[(?P<da>[\d:]+),\s*"
    r"(?P<toc>[\d.]+)s/it(?:,\s*step size=(?P<buoc>[\d.e+-]+))?(?:,\s*acc\. prob=(?P<acc>[\d.]+))?"
)


def _dong_cuoi(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").strip().split("\n")[-1]
    except Exception:
        return ""


def _moi_nhat(mau: str):
    ds = sorted(LOGS.glob(mau), key=lambda p: p.stat().st_mtime, reverse=True)
    return ds[0] if ds else None


def _hhmm(giay: float) -> str:
    if giay < 0 or giay != giay:
        return "--"
    return f"{int(giay // 3600)}h{int((giay % 3600) // 60):02d}m"


def phan_mcmc() -> list[str]:
    L = []
    tong_con_lai = 0.0
    for c in range(4):
        f = _moi_nhat(f"mcmcchain_worker-mcmc-chain_{c}_*.log")
        if f is None:
            f = _moi_nhat(f"mcmcchain_--worker-mcmc-chain_{c}_*.log")
        if f is None:
            continue
        dong = _dong_cuoi(f)
        if "xong sau" in dong:
            L.append(f"  chuoi {c}  XONG")
            continue
        m = _TQDM.search(dong)
        if not m:
            L.append(f"  chuoi {c}  (chua co so lieu)")
            continue
        i, n = int(m["i"]), int(m["n"])
        toc = float(m["toc"])
        con_lai = (n - i) * toc
        tong_con_lai = max(tong_con_lai, con_lai)
        buoc = m["buoc"] or "--"
        acc = m["acc"] or "--"
        thanh = "#" * int(20 * i / n) + "." * (20 - int(20 * i / n))
        L.append(f"  chuoi {c}  [{thanh}] {i:>5}/{n} {m['pha']:<7} "
                 f"{toc:6.1f}s/vong  buoc={buoc:<9} acc={acc:<6} con ~{_hhmm(con_lai)}")
    if L:
        moc = datetime.now() + timedelta(seconds=tong_con_lai)
        L.append(f"  => du kien xong: {moc:%d/%m %H:%M}")
    return L


def phan_mo_phong() -> list[str]:
    sdir = RESULTS / "parts" / "sim"
    if not sdir.exists():
        return []
    can = {"mle": 30, "vb_mean_field": 15, "mcmc": 3}
    codes = [f"S{i}" for i in range(1, 9)]
    L = [f"  {'':<4}{'MLE':>12}{'VB':>12}{'MCMC':>10}"]
    tong_xong = tong_can = 0
    for code in codes:
        o = []
        for m, k in can.items():
            n = len(list(sdir.glob(f"{code}_{m}_*.pt")))
            tong_xong += n
            tong_can += k
            o.append(f"{n}/{k}")
        L.append(f"  {code:<4}{o[0]:>12}{o[1]:>12}{o[2]:>10}")
    # toc do tu cac tep da xong
    tep = sorted(sdir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    toc_do = ""
    if len(tep) >= 2:
        khoang = tep[-1].stat().st_mtime - tep[0].stat().st_mtime
        moi_lan = khoang / max(len(tep) - 1, 1)
        con = tong_can - tong_xong
        toc_do = f"  ~{moi_lan/60:.0f} phut/lan lap, con {con} lan => ~{_hhmm(con*moi_lan/4)}"
    thanh = "#" * int(30 * tong_xong / max(tong_can, 1)) + "." * (30 - int(30 * tong_xong / max(tong_can, 1)))
    L.append(f"  [{thanh}] {tong_xong}/{tong_can} lan lap{toc_do}")
    dang = _dong_cuoi(LOGS / "mo_phong.log") if (LOGS / "mo_phong.log").exists() else ""
    if dang:
        L.append(f"  moi nhat: {dang[:100]}")
    return L


def phan_checkpoint() -> list[str]:
    muc = [("00_data", "Du lieu"), ("01_preliminary_tests", "Kiem dinh so bo"),
           ("02_mle", "MLE B1-B4, M1"), ("03_vb", "VB M2, M3"),
           ("04_mcmc", "MCMC M4"), ("05_r1", "R1 nu theo trang thai"),
           ("06_oos", "Danh gia ngoai mau")]
    L = []
    for ten, mo_ta in muc:
        p = RESULTS / f"{ten}.pt"
        dau = "[x]" if p.exists() else "[ ]"
        kt = f"  {p.stat().st_size/1024:.0f} KB" if p.exists() else ""
        L.append(f"  {dau} {mo_ta:<26}{kt}")
    n = len(list(RESULTS.glob("07_simulation_*.pt")))
    L.append(f"  {'[x]' if n == 8 else '[ ]'} {'Mo phong S1-S8':<26}  {n}/8 kich ban")
    return L


def ve() -> str:
    K = []
    K.append("=" * 78)
    K.append(f"  TIEN DO  --  {datetime.now():%d/%m/%Y %H:%M:%S}")
    K.append("=" * 78)
    mc = phan_mcmc()
    if mc:
        K.append("\nMCMC (NUTS) -- 4 chuoi song song")
        K.append("-" * 78)
        K += mc
    mp = phan_mo_phong()
    if mp:
        K.append("\nMO PHONG S1-S8")
        K.append("-" * 78)
        K += mp
    K.append("\nCHECKPOINT")
    K.append("-" * 78)
    K += phan_checkpoint()
    K.append("")
    return "\n".join(K)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lap", action="store_true", help="Cap nhat lien tuc")
    ap.add_argument("--moi", type=int, default=30, help="So giay giua hai lan cap nhat")
    args = ap.parse_args()
    if not args.lap:
        print(ve())
        return 0
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(ve())
            print(f"  (cap nhat moi {args.moi}s -- Ctrl+C de thoat)")
            time.sleep(args.moi)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
