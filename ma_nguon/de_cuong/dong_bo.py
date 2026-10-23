"""Đồng bộ kết quả giữa các máy qua git (máy chủ trì và các máy chỉ làm đơn vị).

THIẾT KẾ (THIET_KE_CODE.md mục 6c): mỗi tệp trong ket_qua/ chỉ có đúng một máy ghi (đơn vị theo chủ sở hữu, nhật ký
theo máy, hồ sơ/tổng hợp chỉ máy chủ trì), nên `git pull --rebase` không bao giờ gặp xung đột nội dung. Hàm `day`
thử lại nhiều lần khi push bị từ chối (máy khác vừa push): pull --rebase rồi push lại.
Không ghi token ra màn hình: thông báo lỗi của git được cắt bớt và che phần xác thực trong URL.
THƯ VIỆN: subprocess, json, socket.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from .khung import GOC, bam_tep, ghi_nguyen_tu, thong_tin_may


def _che(s: str) -> str:
    return re.sub(r"https://[^@\s]+@", "https://***@", s or "")


def _git(*a, timeout=900) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=GOC, capture_output=True, text=True, timeout=timeout)


def commit_hien_tai() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


def bam_ma() -> str:
    """Băm mã nguồn, cấu hình và dữ liệu đầu vào: đổi sau khi pull nghĩa là tiến trình đang chạy đã cũ, phải khởi động lại.

    Dữ liệu (du_lieu/goc, du_lieu/ngoai; trừ tệp ghi chú .md) được băm từ 23/10/2026: trước đó sửa dữ liệu (bổ sung giá F2M
    ngày 15/08/2019) không làm máy khởi động lại, nên máy chạy tiếp cả đêm trên dữ liệu cũ."""
    tep = [p for d in ("ma_nguon", "quy_trinh") for p in (GOC / d).rglob("*.py") if "__pycache__" not in p.parts]
    du_lieu = [p for d in ("goc", "ngoai") for p in (GOC / "du_lieu" / d).rglob("*") if p.is_file() and p.suffix != ".md"]
    return bam_tep(tep + du_lieu + [GOC / "cau_hinh" / "de_cuong.toml", GOC / "cau_hinh" / "the_che.toml"])


def keo(log=print) -> bool:
    for i in range(5):
        r = _git("pull", "--rebase", "--autostash")
        if r.returncode == 0:
            return True
        log(f"[git] pull loi (lan {i + 1}): {_che(r.stderr.strip())[:300]}")
        time.sleep(15 * (i + 1))
    return False


def day(may: str, thu_muc_ket_qua: Path, log=print) -> bool:
    """git add ket_qua + commit + push (thử lại khi bị từ chối). Trả về True nếu không có gì để đẩy hoặc đẩy được."""
    try:   # chuẩn hóa tên đầy đủ (Windows có thể trả tên ngắn 8.3 cho cùng một thư mục)
        rel = Path(thu_muc_ket_qua).resolve().relative_to(GOC.resolve())
    except ValueError:
        log(f"[git] LOI: thu muc ket qua {thu_muc_ket_qua} nam ngoai repo {GOC}; khong the dong bo")
        return False
    r = _git("add", "-A", str(rel))
    if r.returncode != 0:
        log(f"[git] LOI git add: {_che(r.stderr.strip())[:300]}")
        return False
    if _git("diff", "--cached", "--quiet").returncode == 0:
        return True
    r = _git("commit", "-q", "-m", f"ket qua {may} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if r.returncode != 0:
        log(f"[git] commit loi: {_che(r.stderr.strip())[:300]}")
        return False
    for i in range(8):
        r = _git("push", "-q")
        if r.returncode == 0:
            log("[git] da push ket qua")
            return True
        log(f"[git] push bi tu choi (lan {i + 1}): {_che(r.stderr.strip())[-300:]}")
        keo(log)
        time.sleep(10 * (i + 1))
    log("[git] KHONG push duoc; ket qua van an toan tren may nay, lan dong bo sau se thu lai")
    return False


def dang_ky(may: str, thu_muc_ket_qua: Path) -> str | None:
    """Ghi tệp đăng ký máy (nhịp tim). Trả về cảnh báo nếu tên máy đang được một máy khác dùng gần đây (< 3 giờ)."""
    p = thu_muc_ket_qua / "_may" / f"{may}.json"
    canh_bao = None
    host = socket.gethostname()
    if p.exists():
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
            if x.get("host") != host and time.time() - float(x.get("nhip", 0)) < 3 * 3600:
                canh_bao = (f"Ten may '{may}' dang duoc host '{x.get('host')}' dung (nhip tim "
                            f"{(time.time() - float(x['nhip'])) / 60:.0f} phut truoc). Hay chon ten/phan manh khac.")
        except (ValueError, OSError):
            pass
    ghi_nguyen_tu(p, json.dumps({"may": may, "host": host, "nhip": time.time(), "pid": os.getpid(),
                                 "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False).encode("utf-8"))
    return canh_bao


def ghi_moi_truong(may: str, thu_muc_ket_qua: Path):
    """Lưu phiên bản thư viện và thông tin máy (Phụ lục C.8, mục 8.5)."""
    pf = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    noi_dung = json.dumps(thong_tin_may(), ensure_ascii=False, indent=1) + "\n\n" + pf
    ghi_nguyen_tu(thu_muc_ket_qua / "_moi_truong" / f"{may}.txt", noi_dung.encode("utf-8"))
