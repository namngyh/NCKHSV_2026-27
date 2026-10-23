"""Điểm vào duy nhất của đường ống (chay.bat và notebook Colab gọi tệp này).

Ví dụ:
  python quy_trinh/chay.py --buoc tat_ca                          một máy làm hết (tiếp tục từ điểm kiểm tra)
  python quy_trinh/chay.py --buoc tat_ca --ho_so nhanh            chạy thử nhanh
  python quy_trinh/chay.py --buoc tat_ca --vai_tro chu_tri --may may_manh --git
  python quy_trinh/chay.py --buoc tat_ca --vai_tro don_vi  --may colab_1 --git --gio_toi_da 11
  python quy_trinh/chay.py --buoc b04 b05                         chạy riêng vài bước (một lượt)
  python quy_trinh/chay.py --tien_do                              bảng tiến độ
  python quy_trinh/chay.py --xoa b05                              xóa hồ sơ, điểm kiểm tra và đầu ra của b05
Vòng lặp (--buoc tat_ca): đi theo chuỗi chính b00 -> b14; mỗi bước chạy trong hộp thời gian `phut_dong_bo` rồi đồng
bộ git; khi chuỗi chính phải chờ (máy khác, máy chủ trì) thì làm việc nền b03b (mô phỏng Bayes, SBC). Khi b03b xong,
máy chủ trì làm lại b13, b15, b14. Chuỗi in ra màn hình không dấu để tránh lỗi mã hóa của cửa sổ lệnh Windows.
Mã thoát: 0 xong/hết việc, 1 lỗi, 3 cổng không đạt, 4 mã nguồn đổi sau khi pull (khởi động lại), 130 dừng tay.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MO_DUN = ["quy_trinh.b00_b02_chuan_bi", "quy_trinh.b03_mo_phong", "quy_trinh.b03b_so_lan_lap", "quy_trinh.b03a_ged",
          "quy_trinh.b04_b07_giai_doan_1",
          "quy_trinh.b08_b09_ngoai_mau", "quy_trinh.b10_b12_ky_quy", "quy_trinh.b13_b14_bao_cao"]
THU_TU = ["b00", "b01", "b02", "b02k", "b03a", "b04", "b04b", "b05", "b06", "b07", "b08", "b08b", "b09", "b10",
          "b11", "b12", "b13", "b15", "b14"]
NEN = "b03b"
CUOI = ("b13", "b15", "b14")


def nap_buoc():
    for m in MO_DUN:
        importlib.import_module(m)


def in_tien_do(K, ho_so):
    hang = {h["ma"]: h for h in K.bang_tien_do(ho_so)}
    print(f"{'buoc':5s} {'ten':28s} {'trang thai':16s} {'don vi':>11s}  ket thuc")
    for ma in THU_TU + [NEN]:
        h = hang.get(ma)
        if h:
            print(f"{ma:5s} {h['ten']:28s} {h['trang_thai']:16s} {h['don_vi']:>11s}  {h['ket_thuc']}")


def vong_lap(a, K, DB) -> int:
    ctx0 = K.NguCanh(a.ho_so)
    if a.vai_tro == "chu_tri" and a.may != ctx0.cfg["phan_tan"]["chu_tri"]:
        print(f"[loi] chi may '{ctx0.cfg['phan_tan']['chu_tri']}' duoc chay vai tro chu_tri (xem [phan_tan] trong cau hinh)")
        return 2
    if a.vai_tro != "mot_minh" and a.may not in ctx0.cfg["phan_tan"]["may"]:
        print(f"[loi] ten may '{a.may}' khong co trong [phan_tan.may]")
        return 2
    if a.gio_toi_da:
        os.environ["NCKHSV_HAN"] = str(time.time() + a.gio_toi_da * 3600)
    chu_ky = (a.phut_dong_bo or ctx0.cfg["phan_tan"]["phut_dong_bo"]) * 60
    may = a.may if a.vai_tro != "mot_minh" else None
    ten_may = a.may or "mot_minh"
    if a.git:
        DB.keo()
        canh = DB.dang_ky(ten_may, ctx0.ket_qua)
        if canh:
            print("[canh bao]", canh)
    DB.ghi_moi_truong(ten_may, ctx0.ket_qua)
    ma0 = DB.bam_ma()
    kw = dict(ho_so=a.ho_so, chap_nhan_cong=a.chap_nhan_cong, so_tien_trinh=a.so_tien_trinh, may=may, vai_tro=a.vai_tro)
    bat_dau = time.time()
    nguong = ctx0.cfg["phan_tan"].get("nhan_thay_sau_gio", 6) * 3600

    def may_vang() -> tuple:
        """Máy khác không gửi nhịp tim quá ngưỡng (hoặc chưa từng đăng ký sau khi chủ trì đã chạy quá ngưỡng)."""
        if a.vai_tro != "chu_tri":
            return ()
        import json
        vang = []
        for m in ctx0.cfg["phan_tan"]["may"]:
            if m == a.may:
                continue
            p = ctx0.ket_qua / "_may" / f"{m}.json"
            try:
                nhip = float(json.loads(p.read_text(encoding="utf-8"))["nhip"]) if p.exists() else None
            except (ValueError, OSError, KeyError):
                nhip = None
            if (nhip is not None and time.time() - nhip > nguong) or (nhip is None and time.time() - bat_dau > nguong):
                vang.append(m)
        if vang:
            print(f"[nhan thay] may vang mat qua {nguong / 3600:.0f} gio: {vang}; may chu tri lam thay phan viec cua chung")
        return tuple(vang)

    while K.con_thoi_gian():
        kw["nhan_thay"] = may_vang()
        if a.git:
            DB.keo()
            if DB.bam_ma() != ma0:
                print("[dung] Ma nguon/cau hinh tren GitHub da thay doi. Hay khoi dong lai de dung ban moi.")
                DB.day(ten_may, ctx0.ket_qua)
                return 4
            DB.dang_ky(ten_may, ctx0.ket_qua)
        con_viec_chinh, chan, chinh_xong = False, None, True
        for ma in THU_TU:
            st = K.chay_buoc(ma, han=time.time() + chu_ky, **kw)
            if st in ("xong", "bo_qua", "bo_qua_chu_tri", "xong_phan_minh"):
                continue
            chinh_xong = False
            if st == "cong_khong_dat":
                if a.git:
                    DB.day(ten_may, ctx0.ket_qua)
                return 3
            if st == "het_gio":
                con_viec_chinh = True
            else:
                chan = (ma, st)
            break
        if a.git:
            DB.day(ten_may, ctx0.ket_qua)
        if con_viec_chinh:
            continue
        st_nen = K.chay_buoc(NEN, han=time.time() + chu_ky, **kw)
        if st_nen == "xong" and a.vai_tro != "don_vi":
            for ma in CUOI:
                if (K.doc_ho_so(ctx0, ma) or {}).get("trang_thai") == "xong":
                    K.chay_buoc(ma, chay_lai=True, **kw)
        if a.git:
            DB.day(ten_may, ctx0.ket_qua)
        if st_nen == "het_gio":
            continue
        if chinh_xong and st_nen in ("xong", "bo_qua"):
            print("HOAN TAT: moi buoc da xong.")
            return 0
        if a.vai_tro == "don_vi":
            print(f"Het viec cua may '{a.may}' luc nay (cho may chu tri: {chan}). Chay lai sau de nhan viec moi.")
            return 0
        if a.vai_tro == "mot_minh":
            print(f"[dung] chuoi chinh bi chan tai {chan}")
            return 1
        print(f"Cho may khac/tong hop ({chan}); kiem tra lai sau {min(600, chu_ky) // 60} phut")
        time.sleep(min(600, chu_ky))
    print("Het gio toi da: da ghi diem kiem tra; chay lai se tiep tuc.")
    if a.git:
        DB.day(ten_may, ctx0.ket_qua)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ho_so", default="day_du", choices=["day_du", "nhanh"])
    ap.add_argument("--buoc", nargs="*", default=[])
    ap.add_argument("--tien_do", action="store_true")
    ap.add_argument("--chay_lai", action="store_true", help="bo qua kiem tra 'da xong' (diem kiem tra van duoc dung lai)")
    ap.add_argument("--xoa", nargs="*", default=[])
    ap.add_argument("--chap_nhan_cong", action="store_true", help="chu dong chay tiep khi cong khong dat (ghi vao ho so)")
    ap.add_argument("--so_tien_trinh", type=int, default=None)
    ap.add_argument("--vai_tro", default="mot_minh", choices=["mot_minh", "chu_tri", "don_vi"])
    ap.add_argument("--may", default=None, help="ten may trong [phan_tan.may]")
    ap.add_argument("--gio_toi_da", type=float, default=None, help="dung sach sau so gio nay (Colab: 11)")
    ap.add_argument("--phut_dong_bo", type=int, default=None)
    ap.add_argument("--git", action="store_true", help="tu dong git pull/push ket qua")
    a = ap.parse_args()
    if a.vai_tro != "mot_minh" and not a.may:
        ap.error("--vai_tro chu_tri/don_vi can --may")
    nap_buoc()
    from de_cuong import dong_bo as DB
    from de_cuong import khung as K
    for ma in a.xoa:
        K.xoa_buoc(ma, a.ho_so)
        print(f"Da xoa ho so, diem kiem tra va dau ra cua {ma} ({a.ho_so})")
    if a.tien_do:
        in_tien_do(K, a.ho_so)
    try:
        if a.buoc == ["tat_ca"]:
            try:
                return vong_lap(a, K, DB)
            finally:
                if a.git:      # kể cả khi gặp lỗi bất ngờ: đẩy những đơn vị đã xong để máy khác thấy
                    DB.day(a.may or "mot_minh", K.NguCanh(a.ho_so).ket_qua)
        for ma in a.buoc:
            if ma not in K.DANH_MUC:
                print(f"[loi] khong co buoc {ma}")
                return 2
            st = K.chay_buoc(ma, a.ho_so, chay_lai=a.chay_lai, chap_nhan_cong=a.chap_nhan_cong,
                             so_tien_trinh=a.so_tien_trinh, may=a.may if a.vai_tro != "mot_minh" else None,
                             vai_tro=a.vai_tro)
            print(f"{ma}: {st}")
            if st == "cong_khong_dat":
                return 3
    except KeyboardInterrupt:
        print("\nDa dung theo yeu cau. Chay lai se tiep tuc tu diem kiem tra.")
        return 130
    except Exception as e:      # noqa: BLE001
        traceback.print_exc()
        print(f"[loi] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
