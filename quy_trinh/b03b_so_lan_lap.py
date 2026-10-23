"""b03b: số lần lặp riêng cho VB (B_vb) và NUTS (B_nuts) của mô phỏng khôi phục tham số Bayes (22/10/2026).

LÝ THUYẾT: mỗi lần lặp b của trường hợp (phân phối, E) có hạt giống dữ liệu riêng (dùng chung với b03a), nên kết quả của
một lần lặp không phụ thuộc tổng số lần lặp. Vì vậy:
  - B_vb, B_nuts nằm ở mục cấu hình [mo_phong_bayes], KHÔNG đưa vào mã băm đầu vào của b03b;
  - b03b được đánh dấu `mo_rong_don_vi`: tăng B (vd. B_nuts 300 -> 1000) chỉ chạy thêm các lần lặp còn thiếu, các lần
    lặp đã xong được dùng lại; giảm B thì chỉ các lần lặp đầu được tổng hợp.
Sai số mô phỏng của độ phủ khoảng 95% là sqrt(0,95 x 0,05 / B): 0,007 với B = 1000, 0,013 với B = 300.
Đặt ở tệp riêng (không sửa b03_mo_phong.py) vì b03a băm cả tệp b03_mo_phong.py: sửa tệp đó sẽ làm b03a (đã xong 12.000
đơn vị) chạy lại từ đầu.
Từ 23/10/2026 b03b chạy cả GED khi GED tham gia chọn phân phối (bám BM02), giống b03a (quy_trinh/b03a_ged.py): thêm
29.800 đơn vị (VB-MF, VB-FR, NUTS và SBC của GED); tổng hợp `_th_b03b` đã tổng quát theo phân phối nên không cần đổi.
THƯ VIỆN: không (chỉ sinh danh sách đơn vị).
"""
from __future__ import annotations

import quy_trinh.b03_mo_phong as B3
from de_cuong import mo_phong as MP
from de_cuong.khung import DANH_MUC
from quy_trinh.b03a_ged import cac_phan_phoi


def so_lan_lap(cfg: dict, pt: str) -> int:
    """Số lần lặp của phương pháp pt; thiếu [mo_phong_bayes] thì dùng [mo_phong].B như trước."""
    mb = cfg.get("mo_phong_bayes", {})
    return int(mb.get("B_nuts" if pt == "nuts" else "B_vb", cfg["mo_phong"]["B"]))


def _dv_b03b(ctx):
    c = ctx.cfg
    ds = []
    for pp in cac_phan_phoi(c):
        for pt in c["bayes"]["phuong_phap"]:
            ds += [f"mpb|{pp}|{pt}|{kb}|{b:03d}|{h}::0" for kb in MP.KICH_BAN for b in range(so_lan_lap(c, pt))
                   for h in B3._hau_to(ctx, pt)]
            ds += [f"sbc|{pp}|{pt}|{l:03d}|{h}::0" for l in range(c["mo_phong"]["sbc_so_lan"]) for h in B3._hau_to(ctx, pt)]
    return ds


DANH_MUC["b03b"].don_vi = _dv_b03b
DANH_MUC["b03b"].mo_rong_don_vi = True
