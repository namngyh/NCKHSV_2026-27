"""Kiểm tra mã băm đầu vào của b03a và b03b trên máy này có khớp máy chủ trì không.

LÝ THUYẾT: Mỗi bước được nhận diện bằng mã băm của mã nguồn, cấu hình và dữ liệu đầu vào (khung._bam_vao). Nếu một máy
phụ có tệp mã khác byte (ví dụ xuống dòng LF thay vì CRLF) thì mã băm lệch, máy đó sẽ tính vào một thư mục khác và kết
quả không được ghép. Kiểm tra này chạy trước khi máy phụ bắt đầu, để dừng sớm khi lệch.
THƯ VIỆN: Chỉ dùng khung và chay của dự án; không tính toán, không ghi tệp.
Tham số: sys.argv[1] (tùy chọn) = tên máy.
Trả về: mã thoát 0 nếu khớp, 2 nếu lệch.
Tệp này không nằm trong danh sách tệp mã của bước nào nên thêm hay sửa nó không đổi mã băm.
"""
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
for p in (GOC, GOC / "ma_nguon", GOC / "quy_trinh"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from de_cuong import khung as K  # noqa: E402
import chay  # noqa: E402

KY_VONG = {"b03a": "64b56de0015e", "b03b": "14e4a097af61"}

chay.nap_buoc()
may = sys.argv[1] if len(sys.argv) > 1 else None
lech = False
for ma, mong in KY_VONG.items():
    b = K.DANH_MUC[ma]
    h = K._bam_vao(K.NguCanh("day_du", b, may), b)[:12]
    ok = h == mong
    lech |= not ok
    print(f"{ma}: {h}  (can {mong})  {'KHOP' if ok else 'LECH'}")
sys.exit(2 if lech else 0)
