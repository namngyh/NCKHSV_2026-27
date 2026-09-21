"""Bang theo doi tien do rieng cho HAI nghien cuu khoi phuc tham so.

Chi DOC checkpoint tren dia, khong gan vao tien trinh nao, nen chay duoc bat cu
luc nao va bao nhieu cua so cung duoc, khong anh huong gi den lan chay dang co.

    PR-A  MS-GARCH-X Student-t   S1,S2,S3   -> results/parts/sim/<KB>_<pp>_<b>.pt
    PR-B  EGARCH-X Student-t     E1,E2,E3   -> results/mo_phong_egarch/<KB>.pt

Chay:
    python scripts/tien_ich/tien_do_mo_phong.py
    python scripts/tien_ich/tien_do_mo_phong.py --lap          # cap nhat moi 60 giay
    python scripts/tien_ich/tien_do_mo_phong.py --jobs 12      # de uoc luong thoi gian con lai
Hoac double-click `chay_mo_phong.bat` roi chon muc "Xem tien do".
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

SIM_DIR = ROOT / "results" / "parts" / "sim"
EGARCH_DIR = ROOT / "results" / "mo_phong_egarch"

KICH_BAN_MS = ["S1", "S2", "S3"]
KICH_BAN_E = ["E1", "E2", "E3"]
PHUONG_PHAP = ["mle", "vb_mean_field", "vb_full_rank"]


def _doc_elapsed(path: Path) -> float | None:
    """Doc thoi gian chay cua mot lan lap. Tra ve None neu file hong.

    File hong khong phai chuyen gia dinh: neu may tat dien dung luc dang ghi
    thi file .pt co the cut doi. Bat loi o day de bang tien do van in duoc,
    va de bao cho nguoi dung biet file nao can xoa di chay lai.
    """
    try:
        return float(torch.load(path, weights_only=False)["elapsed"])
    except Exception:
        return None


def tien_do_pr_a(B: int, jobs: int) -> None:
    print("=" * 74)
    print(f"PR-A  MS-GARCH-X Student-t (2 trang thai, 15 tham so)   muc tieu B={B}, n=2000")
    print("=" * 74)
    if not SIM_DIR.exists():
        print("  (chua co thu muc results/parts/sim -- chua chay lan nao)")
        return

    print(f"  {'kich ban / phuong phap':<34}{'xong':>7}{'con':>7}{'TB (phut)':>12}")
    tong_xong = tong_con = 0
    gio_con_lai = 0.0
    con_chua_do = 0          # lan lap con lai o nhung o CHUA co mot mau thoi gian nao
    hong: list[str] = []

    for kb in KICH_BAN_MS:
        for pp in PHUONG_PHAP:
            files = sorted(SIM_DIR.glob(f"{kb}_{pp}_*.pt"))
            times = []
            for f in files:
                t = _doc_elapsed(f)
                if t is None:
                    hong.append(f.name)
                else:
                    times.append(t / 60.0)
            xong, con = len(files), max(B - len(files), 0)
            tb = st.mean(times) if times else float("nan")
            tong_xong += xong
            tong_con += con
            if times:
                gio_con_lai += con * tb / 60.0
            else:
                con_chua_do += con
            tb_txt = f"{tb:>12.1f}" if times else f"{'--':>12}"
            print(f"  {kb + ' / ' + pp:<34}{xong:>7}{con:>7}{tb_txt}")

    print("-" * 74)
    tong_muc_tieu = len(KICH_BAN_MS) * len(PHUONG_PHAP) * B
    pct = 100.0 * tong_xong / tong_muc_tieu if tong_muc_tieu else 0.0
    print(f"  {'TONG':<34}{tong_xong:>7}{tong_con:>7}   ({pct:.1f}% cua {tong_muc_tieu})")
    if tong_con and gio_con_lai:
        print(f"  Da do duoc: {gio_con_lai:.0f} gio gop / {jobs} tien trinh "
              f"= ~{gio_con_lai / jobs:.1f} gio")
    # Khong gop cac o chua co mau thoi gian vao con so tren: lam vay se cho ra
    # mot ETA nho hon su that ma khong noi ro vi sao. Bao rieng de nguoi doc
    # biet con so kia CHUA phai tong.
    if con_chua_do:
        print(f"  CHUA TINH DUOC: {con_chua_do} lan lap o nhung o chua co mot mau "
              f"thoi gian nao (cot 'TB' la '--').")
        print("  => con so gio o tren la CAN DUOI, khong phai tong thoi gian con lai.")
    if hong:
        print(f"  !! {len(hong)} file HONG (doc khong duoc) -- xoa di roi chay lai:")
        for ten in hong[:10]:
            print(f"       results/parts/sim/{ten}")
    print()


def tien_do_pr_b() -> None:
    print("=" * 74)
    print("PR-B  EGARCH-X Student-t (1 trang thai, 8 tham so)      muc tieu B=100, n=2000")
    print("=" * 74)
    if not EGARCH_DIR.exists():
        print("  (chua chay lan nao)\n")
        return
    for kb in KICH_BAN_E:
        path = EGARCH_DIR / f"{kb}.pt"
        if not path.exists():
            print(f"  {kb:<8}chua xong")
            continue
        try:
            d = torch.load(path, weights_only=False)
            giay = d.get("giay")
            n = next((d[pp]["n_replications"] for pp in PHUONG_PHAP if pp in d), "?")
            gio = f"{giay / 3600:.1f} gio" if giay else "--"
            print(f"  {kb:<8}XONG   {n} lan lap/phuong phap   ({gio})")
        except Exception:
            print(f"  {kb:<8}!! FILE HONG -- xoa results/mo_phong_egarch/{kb}.pt roi chay lai")
    print()
    n_parts = len(list((EGARCH_DIR / "parts").glob("*.pt"))) if (EGARCH_DIR / "parts").exists() else 0
    print(f"  Checkpoint tung lan lap: {n_parts}/{3 * len(KICH_BAN_E) * 100} "
          f"(trong {EGARCH_DIR.name}/parts/)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lap", action="store_true", help="cap nhat lien tuc")
    ap.add_argument("--moi", type=int, default=60, help="giay giua hai lan cap nhat")
    ap.add_argument("--B", type=int, default=100, help="muc tieu so lan lap cua PR-A")
    ap.add_argument("--jobs", type=int, default=12, help="so tien trinh song song, de uoc tinh")
    args = ap.parse_args()

    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]  TIEN DO KHOI PHUC THAM SO\n")
        tien_do_pr_a(args.B, args.jobs)
        tien_do_pr_b()
        if not args.lap:
            return
        time.sleep(args.moi)


if __name__ == "__main__":
    main()
