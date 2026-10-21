"""Khung đường ống: cấu hình theo hồ sơ, điểm kiểm tra ba tầng, khóa khối kiểm tra, cổng, chạy phân tán.

THIẾT KẾ (THIET_KE_CODE.md mục 1, 3, 6c):
  - Mỗi bước (Buoc) khai báo đơn vị công việc dạng "nhóm::khóa". Các nhóm độc lập (chạy song song), các khóa trong
    một nhóm chạy tuần tự (khởi tạo ấm từ đơn vị trước). Mỗi đơn vị xong được ghi nguyên tử vào
    ket_qua/_diem_kiem_tra/<bước>/<băm đầu vào>/<đơn vị>.pkl (tầng 2); đơn vị dài (chuỗi NUTS, VB) còn ghi trạng
    thái giữa chừng vào thư mục _doan/<đơn vị>/ (tầng 3). Hồ sơ bước (tầng 1) ở ket_qua/_ho_so/<bước>.json.
  - Băm đầu vào = cấu hình liên quan + băm đầu ra của bước phụ thuộc + băm mã nguồn của bước + băm dữ liệu gốc.
    Hai máy chạy hai bản mã khác nhau sẽ ghi vào hai thư mục điểm kiểm tra khác nhau, không thể trộn lẫn.
  - Chạy phân tán: mỗi NHÓM đơn vị có đúng một máy chủ sở hữu, xác định tĩnh bằng băm tên nhóm trên danh sách phân
    mảnh trong [phan_tan] (nhóm NUTS chỉ giao cho các máy trong `may_nuts`). Vai trò:
        mot_minh : một máy làm mọi đơn vị và mọi phần tổng hợp (như khi chạy trên một máy);
        chu_tri  : làm đơn vị của mình; khi đủ đơn vị của mọi máy thì tổng hợp, kiểm tra cổng, ghi hồ sơ;
        don_vi   : chỉ làm đơn vị của mình; không bao giờ ghi hồ sơ, tổng hợp hay khóa.
    Mỗi tệp chỉ có một máy ghi, nên đồng bộ qua git không bao giờ xung đột.
  - Khóa khối kiểm tra (Phụ lục C.7): trước khi có KHOA_CAU_HINH.json, NguCanh.du_lieu() cắt bảng ngày tại hết khối XT.
  - Cổng (mục 8.1): bước có `cong` chỉ chạy khi cổng đạt, trừ khi người dùng chủ động chấp nhận (ghi vào hồ sơ).
  - Giới hạn giờ: biến môi trường NCKHSV_HAN (giây kỷ nguyên). Hết giờ thì đơn vị dài ghi điểm kiểm tra rồi dừng.

THƯ VIỆN: tomllib, hashlib, pickle, concurrent.futures (spawn), subprocess (mã commit), platform.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

GOC = Path(__file__).resolve().parents[2]
THU_VIEN = Path(__file__).resolve().parent
DANH_MUC: dict[str, "Buoc"] = {}
THONG_KE = {"don_vi_moi": 0}


class HetGio(Exception):
    """Hết giới hạn giờ: đơn vị dài đã ghi điểm kiểm tra và dừng sạch."""


def con_thoi_gian() -> bool:
    han = os.environ.get("NCKHSV_HAN")
    return han is None or time.time() < float(han)


# ----------------------------------------------------------------------------------------
# Cấu hình, băm, ghi tệp
# ----------------------------------------------------------------------------------------
def doc_cau_hinh(ho_so: str = "day_du") -> dict:
    """Đọc cau_hinh/de_cuong.toml, áp dụng ghi đè của hồ sơ, gắn Bảng 3 từ the_che.toml."""
    cfg = tomllib.loads((GOC / "cau_hinh" / "de_cuong.toml").read_text(encoding="utf-8"))
    ghi_de = cfg.pop("ho_so", {}).get(ho_so, {})
    if ho_so != "day_du" and not ghi_de:
        raise ValueError(f"Khong co ho so '{ho_so}' trong de_cuong.toml")
    for k, v in ghi_de.items():
        muc, ten = k.split(".", 1)
        cfg[muc][ten] = v
    cfg["the_che"] = tomllib.loads((GOC / "cau_hinh" / "the_che.toml").read_text(encoding="utf-8"))["thong_so"]
    cfg["ten_ho_so"] = ho_so
    return cfg


def js(o):
    """Chuyển đối tượng numpy sang kiểu JSON (NaN/inf -> null)."""
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, np.ndarray):
        return js(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def bam(obj) -> str:
    return hashlib.sha256(json.dumps(js(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bam_tep(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.exists() else b"<khong_co>")
    return h.hexdigest()


_BAM_DU_LIEU: dict[str, str] = {}


def bam_du_lieu(cfg) -> str:
    p = str(GOC / cfg["chung"]["du_lieu_goc"])
    if p not in _BAM_DU_LIEU:
        _BAM_DU_LIEU[p] = bam_tep([p])
    return _BAM_DU_LIEU[p]


def ghi_nguyen_tu(path: Path, du_lieu: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tam = path.with_name(path.name + f".tam{os.getpid()}")
    tam.write_bytes(du_lieu)
    for i in range(10):     # Windows: OneDrive, trình diệt virus hoặc trình chỉ mục có thể giữ tệp trong giây lát
        try:
            os.replace(tam, path)
            return
        except PermissionError:
            if i == 9:
                raise
            time.sleep(0.5 * (i + 1))


def ten_tep(khoa: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", khoa)


def commit_git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=GOC, capture_output=True, text=True,
                              timeout=20).stdout.strip() or "khong_ro"
    except (OSError, subprocess.SubprocessError):
        return "khong_ro"


def ram_trong_gb() -> float | None:
    """RAM còn trống (GB) trên Windows hoặc Linux; None nếu không đọc được."""
    try:
        if os.name == "nt":
            import ctypes

            class _M(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _M()
            m.dwLength = ctypes.sizeof(_M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return min(m.ullAvailPhys, m.ullAvailPageFile) / 2 ** 30
        with open("/proc/meminfo") as f:
            for dong in f:
                if dong.startswith("MemAvailable:"):
                    return int(dong.split()[1]) / 2 ** 20
    except (OSError, AttributeError, ValueError):
        return None
    return None


def so_tien_trinh_tu_dong(cfg) -> int:
    """min(số lõi - 1, RAM trống / bộ nhớ ước tính mỗi tiến trình), giảm một nửa sau mỗi lần tiến trình con chết."""
    n = cfg["chung"]["so_tien_trinh"] or max(1, (os.cpu_count() or 2) - 1)
    ram = ram_trong_gb()
    if ram is not None and not cfg["chung"]["so_tien_trinh"]:
        n = min(n, max(1, int(ram // cfg["chung"].get("bo_nho_moi_tien_trinh_gb", 1.5))))
    return max(1, n >> THONG_KE.get("giam", 0))


def thong_tin_may() -> dict:
    return {"host": socket.gethostname(), "he_dieu_hanh": platform.platform(), "python": sys.version.split()[0],
            "so_loi": os.cpu_count(), "bo_xu_ly": platform.processor()}


# ----------------------------------------------------------------------------------------
# Phân tán: chủ sở hữu của nhóm đơn vị
# ----------------------------------------------------------------------------------------
def la_nhom_nuts(nhom: str) -> bool:
    return "nuts" in nhom.split("|")


def chu_so_huu(nhom: str, cfg: dict) -> str:
    """Máy sở hữu một nhóm đơn vị: băm tên nhóm trên danh sách (phân mảnh, máy) của các máy đủ điều kiện."""
    pt = cfg["phan_tan"]
    du = pt["may_nuts"] if la_nhom_nuts(nhom) else list(pt["may"])
    kho = sorted((s, m) for m in du for s in pt["may"][m])
    h = int(hashlib.sha256(nhom.encode()).hexdigest()[:12], 16)
    return kho[h % len(kho)][1]


# ----------------------------------------------------------------------------------------
# Bước và ngữ cảnh
# ----------------------------------------------------------------------------------------
@dataclass
class Buoc:
    ma: str                               # "b05"
    ten: str                              # "chon_bien"
    muc: str                              # mục đề cương
    tep: str                              # __file__ của mô-đun bước
    lam: Callable                         # lam(ctx, khoa, truoc) -> kết quả đơn vị
    tong_hop: Callable                    # tong_hop(ctx, {khoa: kết quả}) -> None (ghi đầu ra)
    don_vi: Callable = lambda ctx: ["tat_ca::0"]
    phu_thuoc: tuple = ()
    cau_hinh: tuple = ()                  # các mục cấu hình đưa vào băm
    thu_vien: tuple = ()                  # mô-đun de_cuong đưa vào băm
    can_khoa_KT: bool = False             # cần khóa cấu hình, được thấy KT
    truy_cap_KT: bool = False             # được thấy KT mà không cần khóa (chỉ kiểm tra chất lượng dữ liệu)
    song_song: bool = True
    cong: Callable | None = None          # cong(ctx) -> {"dat": bool, ...}
    chi_chu_tri: bool = False             # bước nhỏ/tổng hợp: chỉ máy chủ trì (hoặc một mình) chạy

    def __post_init__(self):
        DANH_MUC[self.ma] = self

    @property
    def thu_muc(self) -> str:
        return f"{self.ma}_{self.ten}"


class NguCanh:
    def __init__(self, ho_so: str = "day_du", buoc: Buoc | None = None, may: str | None = None):
        self.ho_so = ho_so
        self.cfg = doc_cau_hinh(ho_so)
        goc_kq = os.environ.get("NCKHSV_KET_QUA")
        self.ket_qua = Path(goc_kq) if goc_kq else GOC / ("ket_qua" if ho_so == "day_du" else f"ket_qua_{ho_so}")
        self.buoc = buoc
        self.may = may
        self._d = None
        self._khoi = None
        self._dk = None                   # thư mục điểm kiểm tra của bước đang chạy

    # --- đường dẫn và đọc/ghi ---
    def thu_muc(self, ma: str | None = None) -> Path:
        if ma and ma not in DANH_MUC:     # tiến trình con chỉ nạp mô-đun của bước đang chạy
            ds = sorted(self.ket_qua.glob(f"{ma}_*"))
            if not ds:
                raise RuntimeError(f"Chua co ket qua cua buoc {ma}")
            return ds[0]
        b = DANH_MUC[ma] if ma else self.buoc
        return self.ket_qua / b.thu_muc

    def thu_muc_doan(self, khoa: str) -> Path:
        """Thư mục điểm kiểm tra tầng 3 (trạng thái giữa chừng) của một đơn vị."""
        p = self._dk / "_doan" / ten_tep(khoa)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def ghi(self, ten: str, obj):
        p = self.thu_muc() / ten
        if ten.endswith(".json"):
            ghi_nguyen_tu(p, json.dumps(js(obj), ensure_ascii=False, indent=1).encode("utf-8"))
        elif ten.endswith(".pkl"):
            ghi_nguyen_tu(p, pickle.dumps(obj))
        elif ten.endswith(".csv"):
            ghi_nguyen_tu(p, obj.to_csv(index=False).encode("utf-8"))
        else:
            ghi_nguyen_tu(p, obj if isinstance(obj, bytes) else str(obj).encode("utf-8"))
        return p

    def doc(self, ten: str, ma: str | None = None):
        p = self.thu_muc(ma) / ten
        if ten.endswith(".json"):
            return json.loads(p.read_text(encoding="utf-8"))
        if ten.endswith(".pkl"):
            return pickle.loads(p.read_bytes())
        raise ValueError(ten)

    def co_tep(self, ten: str, ma: str) -> bool:
        try:
            return (self.thu_muc(ma) / ten).exists()
        except RuntimeError:
            return False

    def log(self, msg: str):
        ma = self.buoc.ma if self.buoc else "chung"
        dong = f"[{time.strftime('%H:%M:%S')}] [{ma}] [{self.may or 'mot_minh'}] {msg}"
        print(dong, flush=True)
        p = self.ket_qua / "_nhat_ky" / f"{ma}_{self.may or 'mot_minh'}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(dong + "\n")

    # --- khóa khối kiểm tra ---
    @property
    def tep_khoa(self) -> Path:
        return self.ket_qua / "KHOA_CAU_HINH.json"

    def co_khoa_KT(self) -> bool:
        return self.tep_khoa.exists()

    def khoa(self) -> dict:
        return json.loads(self.tep_khoa.read_text(encoding="utf-8"))

    # --- dữ liệu ---
    def du_lieu(self):
        """Bảng ngày; cắt tại hết khối XT nếu bước hiện tại không được thấy KT (Phụ lục C.7)."""
        if self._d is None:
            from .du_lieu import xay_ngay
            d = xay_ngay(GOC / self.cfg["chung"]["du_lieu_goc"], self.cfg)
            self._khoi = self._tinh_khoi(len(d))
            duoc_thay = self.buoc is not None and (self.buoc.can_khoa_KT or self.buoc.truy_cap_KT)
            if self.buoc is not None and self.buoc.can_khoa_KT and not self.co_khoa_KT():
                raise RuntimeError("Buoc can khoa cau hinh (b07) truoc khi thay khoi KT")
            if not duoc_thay:
                d = d.iloc[:self._khoi["KT"][0]].copy()
            self._d = d
        return self._d

    def _tinh_khoi(self, n: int) -> dict:
        b0 = self.cfg["du_lieu"]["bo_qua_dau"]
        ul = self.cfg["khoi"]["UL"]
        xt = self.cfg.get("khoi_bm02", {}).get("XT", self.cfg["khoi"]["XT"])    # [khoi_bm02] ghi đè (bám BM02, 22/10/2026)
        return {"UL": (b0, b0 + ul), "XT": (b0 + ul, b0 + ul + xt), "KT": (b0 + ul + xt, n)}

    def khoi(self) -> dict:
        if self._khoi is None:
            self.du_lieu()
        return self._khoi

    def rng(self, *khoa) -> np.random.Generator:
        """Hạt giống cố định theo (hạt giống chung, bước, đơn vị): làm lại một đơn vị ở bất kỳ máy nào cũng cùng hạt giống."""
        h = int(bam([self.cfg["chung"]["hat_giong"], self.buoc.ma if self.buoc else "", *khoa])[:12], 16)
        return np.random.default_rng(h)

    def hat_giong(self, *khoa) -> int:
        return int(self.rng(*khoa).integers(2 ** 31 - 1))


# ----------------------------------------------------------------------------------------
# Hồ sơ và chạy bước
# ----------------------------------------------------------------------------------------
def _tep_ho_so(ctx: NguCanh, ma: str) -> Path:
    return ctx.ket_qua / "_ho_so" / f"{ma}.json"


def doc_ho_so(ctx: NguCanh, ma: str) -> dict | None:
    p = _tep_ho_so(ctx, ma)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ghi_ho_so(ctx: NguCanh, ma: str, hs: dict):
    ghi_nguyen_tu(_tep_ho_so(ctx, ma), json.dumps(js(hs), ensure_ascii=False, indent=1).encode("utf-8"))


class ChuaXongPhuThuoc(RuntimeError):
    pass


# b01, b02 chỉ dùng khối UL (kết quả không đổi khi XT đổi), nên mã băm của chúng không tính [khoi_bm02]: nhờ vậy việc
# đổi XT (bám BM02) không làm b02 chạy lại, không đổi mã băm đầu ra của b02 và không làm b03a mất kết quả đã chạy.
BUOC_KHONG_PHU_THUOC_XT = ("b01", "b02")


def _bam_vao(ctx: NguCanh, b: Buoc) -> str:
    dep = {}
    for m in b.phu_thuoc:
        hs = doc_ho_so(ctx, m)
        if not hs or hs.get("trang_thai") != "xong":
            raise ChuaXongPhuThuoc(f"Buoc {b.ma} can buoc {m} xong truoc")
        dep[m] = hs["h_ra"]
    tep = [b.tep] + [THU_VIEN / f"{m}.py" for m in b.thu_vien]      # khung không ảnh hưởng con số nên không băm
    # chỉ băm phần cấu hình ảnh hưởng con số; số tiến trình, bộ nhớ, phân tán không làm bước phải chạy lại
    chon = {s: ({k: ctx.cfg[s][k] for k in ("hat_giong", "du_lieu_goc")} if s == "chung" else ctx.cfg[s])
            for s in b.cau_hinh}
    if "khoi" in chon and "khoi_bm02" in ctx.cfg and b.ma not in BUOC_KHONG_PHU_THUOC_XT:
        chon["khoi"] = {**chon["khoi"], "XT_bm02": ctx.cfg["khoi_bm02"]["XT"]}
    return bam({"ho_so": ctx.ho_so, "cfg": chon, "dep": dep, "tep": bam_tep(tep),
                "du_lieu": bam_du_lieu(ctx.cfg)})


def _khoi_tao_tien_trinh():
    """Mỗi tiến trình con dùng một luồng để tránh tranh chấp lõi (JAX/BLAS)."""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    for p in (GOC / "ma_nguon", GOC):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _chay_nhom(mo_dun: str, ma: str, ho_so: str, khoa_list: list[str], dk: str, may: str | None) -> int:
    """Chạy tuần tự các đơn vị của một nhóm, bỏ qua đơn vị đã có điểm kiểm tra; dừng sạch khi hết giờ."""
    _khoi_tao_tien_trinh()
    importlib.import_module(mo_dun)
    b = DANH_MUC[ma]
    ctx = NguCanh(ho_so, b, may)
    ctx._dk = Path(dk)
    truoc, moi = None, 0
    for khoa in khoa_list:
        p = Path(dk) / f"{ten_tep(khoa)}.pkl"
        if p.exists():
            truoc = pickle.loads(p.read_bytes())
            continue
        if not con_thoi_gian():
            break
        t = time.time()
        try:
            kq = b.lam(ctx, khoa, truoc)
        except HetGio:
            ctx.log(f"het gio trong don vi {khoa}: da ghi diem kiem tra giua chung, se tiep tuc lan sau")
            break
        if isinstance(kq, dict):
            kq["_nguon"] = {"may": may or "mot_minh", "host": socket.gethostname(),
                            "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S"), "giay": round(time.time() - t, 2)}
        ghi_nguyen_tu(p, pickle.dumps(kq))
        truoc, moi = kq, moi + 1
        ctx.log(f"xong don vi {khoa} ({time.time() - t:.1f} s)")
    return moi


def chay_buoc(ma: str, ho_so: str = "day_du", chay_lai: bool = False, chap_nhan_cong: bool = False,
              so_tien_trinh: int | None = None, may: str | None = None, vai_tro: str = "mot_minh",
              han: float | None = None, nhan_thay: tuple = ()) -> str:
    """Chạy một bước. Trả về: xong, bo_qua, cong_khong_dat, cho_don_vi (chủ trì chờ máy khác), xong_phan_minh,
    bo_qua_chu_tri (máy đơn vị gặp bước chỉ chủ trì), cho_buoc_truoc, het_gio."""
    b = DANH_MUC[ma]
    ctx = NguCanh(ho_so, b, may)
    THONG_KE["don_vi_moi"] = 0
    if vai_tro == "don_vi" and b.chi_chu_tri:
        return "bo_qua_chu_tri"
    try:
        h_vao = _bam_vao(ctx, b)
    except ChuaXongPhuThuoc as e:
        ctx.log(str(e))
        return "cho_buoc_truoc"
    hs = doc_ho_so(ctx, ma)
    if (not chay_lai and hs and hs.get("h_vao") == h_vao and hs.get("trang_thai") == "xong"
            and ctx.thu_muc().exists()):
        return "bo_qua"
    if b.can_khoa_KT and not ctx.co_khoa_KT():
        return "cho_buoc_truoc"
    ket_qua_cong = None
    if b.cong is not None:
        if vai_tro == "don_vi":
            return "bo_qua_chu_tri"
        ket_qua_cong = b.cong(ctx)
        ctx.ghi("cong.json", ket_qua_cong)
        if not ket_qua_cong["dat"]:
            if not chap_nhan_cong:
                ghi_ho_so(ctx, ma, {"trang_thai": "cong_khong_dat", "h_vao": h_vao, "cong": ket_qua_cong})
                ctx.log("CONG KHONG DAT -> dung. Xem cong.json; chay lai voi --chap_nhan_cong neu chu dong chap nhan")
                return "cong_khong_dat"
            ctx.log("CONG KHONG DAT nhung nguoi dung chap nhan (ghi vao ho so)")
    don_vi = b.don_vi(ctx)
    nhom: dict[str, list[str]] = {}
    for k in don_vi:
        nhom.setdefault(k.split("::")[0], []).append(k)
    # máy chủ trì làm thay phần việc của máy vắng mặt (nhan_thay), để đường ống không bị kẹt
    cua_toi = {n: ks for n, ks in nhom.items()
               if vai_tro == "mot_minh" or chu_so_huu(n, ctx.cfg) == may
               or (vai_tro == "chu_tri" and chu_so_huu(n, ctx.cfg) in nhan_thay)}
    dk = ctx.ket_qua / "_diem_kiem_tra" / ma / h_vao[:12]
    dk.mkdir(parents=True, exist_ok=True)
    co = lambda k: (dk / f"{ten_tep(k)}.pkl").exists()
    if vai_tro != "don_vi":
        ghi_ho_so(ctx, ma, {"trang_thai": "dang_chay", "h_vao": h_vao, "so_don_vi": len(don_vi),
                            "thu_muc_diem_kiem_tra": str(dk.relative_to(ctx.ket_qua)),
                            "bat_dau": (hs or {}).get("bat_dau") if (hs or {}).get("h_vao") == h_vao
                            else time.strftime("%Y-%m-%d %H:%M:%S")})
    thieu = [n for n, ks in cua_toi.items() if not all(co(k) for k in ks)]
    ctx.log(f"{b.ten} ({b.muc}): {len(don_vi)} don vi / {len(nhom)} nhom; cua may nay {len(cua_toi)} nhom, "
            f"con thieu {len(thieu)} nhom")
    han_cu = os.environ.get("NCKHSV_HAN")
    if han is not None:
        os.environ["NCKHSV_HAN"] = str(min(han, float(han_cu)) if han_cu else han)
    try:
        nmax = so_tien_trinh or so_tien_trinh_tu_dong(ctx.cfg)
        mo_dun = b.lam.__module__
        if b.song_song and nmax > 1 and len(thieu) > 1:
            from concurrent.futures.process import BrokenProcessPool
            try:
                with ProcessPoolExecutor(max_workers=min(nmax, len(thieu)), initializer=_khoi_tao_tien_trinh,
                                         mp_context=mp.get_context("spawn")) as ex:
                    fut = {ex.submit(_chay_nhom, mo_dun, ma, ho_so, nhom[n], str(dk), may): n for n in thieu}
                    for f in as_completed(fut):
                        THONG_KE["don_vi_moi"] += f.result()
            except BrokenProcessPool:
                THONG_KE["giam"] = THONG_KE.get("giam", 0) + 1
                ctx.log(f"TIEN TRINH CON BI DUNG DOT NGOT (thuong do thieu bo nho). Don vi da xong van duoc giu; "
                        f"lan sau dung {so_tien_trinh_tu_dong(ctx.cfg)} tien trinh. Neu lap lai, dat chung.so_tien_trinh "
                        f"nho hon trong cau hinh.")
                return "het_gio"
        else:
            for n in thieu:
                if not con_thoi_gian():
                    break
                THONG_KE["don_vi_moi"] += _chay_nhom(mo_dun, ma, ho_so, nhom[n], str(dk), may)
    finally:
        if han is not None:
            if han_cu is None:
                os.environ.pop("NCKHSV_HAN", None)
            else:
                os.environ["NCKHSV_HAN"] = han_cu
    if any(not co(k) for ks in cua_toi.values() for k in ks):
        return "het_gio"
    if vai_tro == "don_vi":
        return "xong_phan_minh"
    con_thieu = [k for k in don_vi if not co(k)]
    if con_thieu:
        theo_may: dict[str, int] = {}
        for k in con_thieu:
            m = chu_so_huu(k.split("::")[0], ctx.cfg)
            theo_may[m] = theo_may.get(m, 0) + 1
        ctx.log(f"cho may khac: con thieu {len(con_thieu)} don vi {theo_may}")
        return "cho_don_vi"
    t0 = time.time()
    ket_qua = {k: pickle.loads((dk / f"{ten_tep(k)}.pkl").read_bytes()) for k in don_vi}
    ctx.thu_muc().mkdir(parents=True, exist_ok=True)
    b.tong_hop(ctx, ket_qua)
    theo_may = {}
    for v in ket_qua.values():
        m = v.get("_nguon", {}).get("may", "khong_ro") if isinstance(v, dict) else "khong_ro"
        theo_may[m] = theo_may.get(m, 0) + 1
    ctx.ghi("nguon_goc.json", {"commit": commit_git(), "h_vao": h_vao, "bam_du_lieu": bam_du_lieu(ctx.cfg),
                               "bam_ma": bam_tep([b.tep] + [THU_VIEN / f"{m}.py" for m in b.thu_vien]),
                               "ho_so": ho_so, "may_tong_hop": thong_tin_may(), "so_don_vi_theo_may": theo_may,
                               "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S")})
    h_ra = bam_tep([p for p in ctx.thu_muc().rglob("*") if p.is_file() and p.name != "nguon_goc.json"])
    ghi_ho_so(ctx, ma, {"trang_thai": "xong", "h_vao": h_vao, "h_ra": h_ra, "so_don_vi": len(don_vi),
                        "bat_dau": (doc_ho_so(ctx, ma) or {}).get("bat_dau"),
                        "ket_thuc": time.strftime("%Y-%m-%d %H:%M:%S"), "giay_tong_hop": round(time.time() - t0, 1),
                        "cong": ket_qua_cong, "chap_nhan_cong": bool(chap_nhan_cong and ket_qua_cong
                                                                     and not ket_qua_cong["dat"]),
                        "thu_muc_diem_kiem_tra": str(dk.relative_to(ctx.ket_qua)), "so_don_vi_theo_may": theo_may})
    ctx.log(f"XONG {b.thu_muc}")
    return "xong"


def bang_tien_do(ho_so: str = "day_du") -> list[dict]:
    ctx = NguCanh(ho_so)
    hang = []
    for ma in DANH_MUC:
        b = DANH_MUC[ma]
        hs = doc_ho_so(ctx, ma) or {}
        xong = "-"
        if hs.get("thu_muc_diem_kiem_tra"):
            xong = len(list((ctx.ket_qua / hs["thu_muc_diem_kiem_tra"]).glob("*.pkl")))
        hang.append({"ma": ma, "ten": b.ten, "muc": b.muc, "trang_thai": hs.get("trang_thai", "chua_chay"),
                     "don_vi": f"{xong}/{hs.get('so_don_vi', '-')}", "ket_thuc": hs.get("ket_thuc", "")})
    return hang


def xoa_buoc(ma: str, ho_so: str = "day_du"):
    """Xóa hồ sơ, điểm kiểm tra và đầu ra của một bước (để chạy lại từ đầu)."""
    import shutil
    ctx = NguCanh(ho_so, DANH_MUC[ma])
    for p in (_tep_ho_so(ctx, ma), ctx.ket_qua / "_diem_kiem_tra" / ma, ctx.thu_muc()):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    if ma == "b07" and ctx.tep_khoa.exists():
        ctx.tep_khoa.unlink()
