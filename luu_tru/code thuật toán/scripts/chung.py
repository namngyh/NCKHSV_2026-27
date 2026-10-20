"""Tien ich dung chung cho cac script: duong dan, in nhat ky, luu checkpoint, chay song song."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]          # thu muc "code thuật toán"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import garch_vn30  # noqa: E402,F401  (dat float64 va UTF-8 cho console)

RESULTS = ROOT / "results"
CSV = ROOT / "ohlc_export.csv"
PHUONG_PHAP = ("mle", "vb_mean_field", "vb_full_rank")


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def luu(obj, path: Path) -> None:
    """Luu nguyen tu: ghi file tam roi doi ten, mat dien giua chung khong de lai file cut."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def nap(path: Path):
    return torch.load(path, weights_only=False)


def checkpoint(path: Path, tinh):
    """Da co file thi nap lai; chua co thi goi tinh() va luu (kem thoi gian chay, giay)."""
    if path.exists():
        print(f"    [da co] {path.name}")
        return nap(path)
    t0 = time.time()
    ket_qua = tinh()
    ket_qua["giay"] = time.time() - t0
    luu(ket_qua, path)
    print(f"    [luu] {path.name}  ({ket_qua['giay'] / 60:.1f} phut)")
    return ket_qua


def chay_song_song(tasks: list[list[str]], jobs: int, script: Path, tag: str,
                   args_chung: list[str] | None = None) -> int:
    """Chay moi tac vu thanh mot tien trinh con `python script <args_chung> <task>`,
    toi da `jobs` tien trinh cung luc; nhat ky tung tac vu trong logs/. Tra ve so tac vu loi.

    Moi tien trinh con dung 1 luong BLAS (OMP_NUM_THREADS=1) de nhieu tien trinh
    khong tranh nhau cung mot nhan CPU.
    """
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    hang_doi, dang_chay, loi = list(tasks), [], []
    t0, xong = time.time(), 0
    while hang_doi or dang_chay:
        while hang_doi and len(dang_chay) < jobs:
            task = hang_doi.pop(0)
            ten = "_".join(task[1:])
            fh = open(log_dir / f"{tag}_{ten}_{stamp}.log", "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen([sys.executable, str(script)] + list(args_chung or []) + task,
                                    stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            dang_chay.append((ten, proc, fh))
        time.sleep(5)
        con_chay = []
        for ten, proc, fh in dang_chay:
            rc = proc.poll()
            if rc is None:
                con_chay.append((ten, proc, fh))
                continue
            fh.close()
            xong += 1
            if rc != 0:
                loi.append(ten)
                log(f"  LOI {ten} (ma thoat {rc})")
            if xong % 5 == 0:
                con_lai = len(hang_doi) + len(con_chay)
                tb = (time.time() - t0) / xong
                log(f"  {xong} xong, {con_lai} con lai, uoc tinh con ~{con_lai * tb / 3600:.1f} gio")
        dang_chay = con_chay
    log(f"Ket thuc sau {(time.time() - t0) / 3600:.2f} gio. Loi: {loi or 'khong'}")
    return len(loi)
