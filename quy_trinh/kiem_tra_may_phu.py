"""Kiểm tra một máy phụ trước khi chạy (may_phu.bat gọi tệp này).

LÝ THUYẾT: Máy phụ chỉ làm đơn vị của mình và đẩy lên GitHub; kết quả chỉ dùng được nếu (i) tên máy có trong
[phan_tan.may] và không trùng với máy đang chạy khác (trùng tên thì hai máy làm cùng đơn vị, git xung đột), (ii) mã
băm đầu vào của b03b trên máy này trùng với thư mục điểm kiểm tra đã có. Băm tính trên byte của tệp mã, mà các máy
đang chạy lấy mã với core.autocrlf=true (xuống dòng CRLF); máy lấy mã dạng LF sẽ ra băm khác, ghi vào thư mục khác và
mọi công sức bị bỏ phí.
THƯ VIỆN: tomllib (qua khung), json, socket.
Tham số: tên máy (dòng lệnh).
Trả về: mã thoát 0 đạt; 2 tên máy sai hoặc đang được máy khác dùng; 3 mã băm không khớp.
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(GOC / "ma_nguon"), str(GOC / "quy_trinh"), str(GOC)]


def main() -> int:
    ten = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    import chay
    chay.nap_buoc()
    from de_cuong import khung as K
    cfg = K.NguCanh("day_du").cfg
    pt = cfg["phan_tan"]
    if ten not in pt["may"] or ten == pt["chu_tri"]:
        phu = [m for m in pt["may"] if m != pt["chu_tri"]]
        print(f"[LOI] Ten may '{ten}' khong hop le. Cac ten may phu: {phu}")
        return 2
    p = GOC / "ket_qua" / "_may" / f"{ten}.json"
    if p.exists():
        x = json.loads(p.read_text(encoding="utf-8"))
        tre = (time.time() - float(x.get("nhip", 0))) / 3600
        if x.get("host") != socket.gethostname() and tre < 3:
            print(f"[LOI] Ten '{ten}' dang duoc may '{x.get('host')}' dung ({tre * 60:.0f} phut truoc). Chon ten khac.")
            return 2
    b = K.DANH_MUC["b03b"]
    ctx = K.NguCanh("day_du", b, None)
    h = K._bam_vao(ctx, b)[:12]
    if not (GOC / "ket_qua" / "_diem_kiem_tra" / "b03b" / h).exists():
        print(f"[LOI] Ma bam b03b cua may nay ({h}) khong khop voi ket qua tren GitHub. Thuong do git doi xuong dong:")
        print("      xoa thu muc repo va chay lai may_phu.bat (tep nay tu clone voi core.autocrlf=true, giong may chu tri).")
        return 3
    so = sum(1 for k in b.don_vi(ctx) if K.chu_so_huu(k.split("::")[0], cfg) == ten)
    print(f"[OK] May '{ten}' (host {socket.gethostname()}), ma bam b03b {h} khop; may nay phu trach {so} don vi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
