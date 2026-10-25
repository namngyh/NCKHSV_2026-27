"""Hàm dùng chung cho các bước: đường dẫn, mốc ước lượng lại, MLE cuốn chiếu một mốc, ghép dự báo.

LÝ THUYẾT (mục 5.1): tại mỗi mốc s0 (cách nhau `buoc_uoc_luong_lai` phiên) ước lượng lại trên [BO_QUA, s0) với
thống kê chuẩn hóa chỉ dùng dữ liệu đến s0 - 1, rồi dự báo cho [s0, s1) bằng đệ quy với tham số cố định.
THƯ VIỆN: numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from de_cuong import du_lieu as DL          # noqa: E402
from de_cuong import bien_rv as BRV         # noqa: E402
from de_cuong import uoc_luong as U         # noqa: E402
from de_cuong.mo_hinh_jax import DacTa      # noqa: E402

BRV.gan()   # biến bổ sung RV: thêm vào DL.TEN_BIEN và DL.dac_trung (mọi bước nhập _chung đều thấy RV)


def ung_vien(cfg: dict) -> list:
    """Tập biến ngoại sinh ứng viên: [mo_hinh].bien_ung_vien (đề cương) cộng [chon_bien].them_bien_ung_vien (bổ sung)."""
    return list(cfg["mo_hinh"]["bien_ung_vien"]) + [b for b in cfg["chon_bien"].get("them_bien_ung_vien", [])
                                                    if b not in cfg["mo_hinh"]["bien_ung_vien"]]


def ten_tap(bien) -> str:
    return "+".join(bien) or "EGARCH"


def tu_ten_tap(s: str) -> tuple:
    return () if s == "EGARCH" else tuple(s.split("+"))


def cac_moc(ctx, dau: int, cuoi: int) -> list[int]:
    return list(range(dau, cuoi, ctx.cfg["khoi"]["buoc_uoc_luong_lai"]))


def mle_mot_moc(ctx, pp: str, bien, s0: int, s1: int, truoc=None) -> dict:
    """MLE trên [BO_QUA, s0) và dự báo cắm tham số cho [s0, s1)."""
    d = ctx.du_lieu()
    f = DL.dac_trung(d, s0 - 1)
    spec = DacTa(pp, bien)
    dl = U.chuan_bi(f, spec.bien, s0)
    m = U.mle(spec, dl, ctx.cfg["mle"]["so_diem_khoi_tao"], ctx.cfg["mle"]["do_tan_khoi_tao"],
              ctx.rng(pp, ten_tap(bien), s0), xi_am=None if truoc is None else truoc["xi"])
    db = U.du_bao_cam(spec, m["psi"], dl, s0, s1, ctx.cfg["danh_gia"]["alphas"])
    return {"s0": s0, "s1": s1, "psi": m["psi"], "xi": m["xi"], "ll": m["ll"], "so_gan_tot": m["so_gan_tot"],
            "so_diem": m["so_diem"], "do_tan_ll": m["do_tan_ll"], "db": db, "rho_sao": f["rho_sao"][s0:s1],
            "c_rho": f["c_rho"], "ten": spec.ten()}


def ghep(cac_don_vi: list[dict]) -> dict:
    """Ghép dự báo của các mốc theo thứ tự thời gian."""
    cac_don_vi = sorted(cac_don_vi, key=lambda x: x["s0"])
    db0 = cac_don_vi[0]["db"]
    out = {k: np.concatenate([u["db"][k] for u in cac_don_vi]) for k in ("lnh", "h", "mu", "ls", "pit")}
    for k in ("var", "es"):
        out[k] = {a: {b: np.concatenate([u["db"][k][a][b] for u in cac_don_vi]) for b in ("mua", "ban")}
                  for a in db0[k]}
    out["rho_sao"] = np.concatenate([u["rho_sao"] for u in cac_don_vi])
    out["dau"], out["cuoi"] = cac_don_vi[0]["s0"], cac_don_vi[-1]["s1"]
    if "psi" in cac_don_vi[0]:
        out["psi_cac_moc"] = np.array([u["psi"] for u in cac_don_vi])
    out["s0_cac_moc"] = [u["s0"] for u in cac_don_vi]
    return out


def cat(db: dict, a0: int, a1: int) -> dict:
    """Cắt dự báo đã ghép theo chỉ số tương đối [a0, a1)."""
    out = {k: db[k][a0:a1] for k in ("lnh", "h", "mu", "ls", "pit", "rho_sao")}
    for k in ("var", "es"):
        out[k] = {a: {b: v[a0:a1] for b, v in x.items()} for a, x in db[k].items()}
    return out


def thuc_te(d, dau: int, cuoi: int):
    """Lợi suất và cờ khóa biên của các phiên [dau, cuoi)."""
    return d["r"].to_numpy()[dau:cuoi], d["khoa_bien"].to_numpy()[dau:cuoi]
