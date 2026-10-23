"""b03a mở rộng: mô phỏng khôi phục tham số bằng MLE cho GED (23/10/2026).

LÝ THUYẾT: BM02 đưa GED vào bốn phân phối so sánh ([bam_sat_bm02].ged_tham_gia_chon = true), và b06 đã chọn GED ở lần chạy
này; điều kiện (i) của cổng b07 (mục 8.1) đòi hỏi mô phỏng khôi phục tham số cho đúng phân phối được chọn. b03a ban đầu chỉ
chạy ba phân phối của [mo_hinh].phan_phoi, nên cổng báo "phan phoi ged chua co mo phong khoi phuc tham so (b03a)".
Mô-đun này thêm 4 x B_mle đơn vị của GED vào b03a và tính điều kiện (i) cho cả GED. Cách làm giống b03b_so_lan_lap:
  - kết quả mỗi đơn vị chỉ phụ thuộc hạt giống riêng của (phân phối, kịch bản, b), nên b03a được đánh dấu `mo_rong_don_vi`:
    12.000 đơn vị đã xong của ba phân phối được dùng lại, chỉ chạy phần GED còn thiếu rồi tổng hợp lại;
  - đặt ở tệp riêng, không sửa b03_mo_phong.py, vì b03a băm cả tệp đó (sửa sẽ làm b03a chạy lại từ đầu, khoảng 15 giờ).
Hàm `lam` của b03a đã tổng quát theo phân phối (MP.du_lieu_lan, MP.don_vi_mle), nên không cần đổi.
THƯ VIỆN: không (chỉ sinh danh sách đơn vị và tổng hợp).
"""
from __future__ import annotations

import quy_trinh.b03_mo_phong as B3
from de_cuong import mo_phong as MP
from de_cuong.khung import DANH_MUC


def cac_phan_phoi(cfg: dict) -> list[str]:
    """Ba phân phối của [mo_hinh].phan_phoi, cộng GED khi GED tham gia chọn (bám BM02)."""
    pps = list(cfg["mo_hinh"]["phan_phoi"])
    if cfg["bam_sat_bm02"]["ged_tham_gia_chon"]:
        pps.append(cfg["mo_hinh"]["phan_phoi_thay_the"])
    return pps


def _dv_b03a(ctx):
    c = ctx.cfg
    return [f"mp|{pp}|mle|{kb}|{b:03d}::0" for pp in cac_phan_phoi(c) for kb in MP.KICH_BAN
            for b in range(c["mo_phong"]["B_mle"])]


def _th_b03a(ctx, kq):
    """Như B3._th_b03a, nhưng điều kiện (i) tính cho cả GED."""
    nhom, that_bai = {}, {}
    for k, v in kq.items():
        _, pp, _, kb, _ = k.split("::")[0].split("|")
        if "loi" in v:
            that_bai.setdefault(f"{pp}|{kb}", []).append(v["loi"])
        else:
            nhom.setdefault((pp, kb), []).append(v)
    bang = {f"{pp}|mle|{kb}": MP.tong_hop(v, "mle") for (pp, kb), v in nhom.items()}
    cg = ctx.cfg["cong"]
    dieu_kien = {}
    for pp in cac_phan_phoi(ctx.cfg):
        phu = [bang[f"{pp}|mle|{kb}"]["tham_so"][t]["do_phu"] for kb in MP.KICH_BAN for t in B3.THAM_SO_CONG
               if f"{pp}|mle|{kb}" in bang]
        bb = ([bang[f"{pp}|mle|E2"]["bac_bo_theta0"], bang[f"{pp}|mle|E4"]["bac_bo_gamma0"]]
              if f"{pp}|mle|E2" in bang and f"{pp}|mle|E4" in bang else [])
        dieu_kien[pp] = {"do_phu_min": min(phu) if phu else None, "do_phu_max": max(phu) if phu else None,
                         "bac_bo_sai_E2_theta": bb[0] if bb else None, "bac_bo_sai_E4_gamma": bb[1] if bb else None,
                         "so_that_bai": sum(len(v) for k, v in that_bai.items() if k.startswith(pp + "|")),
                         "dat": bool(phu and bb and cg["do_phu_min"] <= min(phu) and max(phu) <= cg["do_phu_max"]
                                     and max(bb) <= cg["bac_bo_sai_max"])}
    ctx.ghi("mo_phong_khoi_phuc.json", {"B": ctx.cfg["mo_phong"]["B_mle"], "T": ctx.cfg["mo_phong"]["T"],
                                        "kich_ban": {k: list(v) for k, v in MP.KICH_BAN.items()}, "bang": bang,
                                        "dieu_kien_8_1_i": dieu_kien, "tham_so_xet_cong": B3.THAM_SO_CONG,
                                        "so_lan_that_bai": {k: len(v) for k, v in that_bai.items()},
                                        "loi_that_bai": that_bai})


DANH_MUC["b03a"].don_vi = _dv_b03a
DANH_MUC["b03a"].tong_hop = _th_b03a
DANH_MUC["b03a"].mo_rong_don_vi = True
