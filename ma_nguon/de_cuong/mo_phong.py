"""Mô phỏng khôi phục tham số (mục 4.4, Bảng 6) và hiệu chuẩn dựa trên mô phỏng (SBC) cho suy diễn Bayes.

LÝ THUYẾT: Sinh T = 2000 quan sát (sau 200 quan sát khởi động) từ EGARCH-X với psi_0 đã biết: mu_0 = 0,05,
omega_0 = 0,04, kappa_0 = 0,15, beta_0 = 0,95, delta_0 = 0; theta_0, gamma_0, nu_0, lambda_0 theo kịch bản E1-E4
(với phân phối chuẩn bỏ nu, lambda; với Student-t bỏ lambda). Biến ngoại sinh: hai AR(1) chuẩn hóa hệ số 0,5.
Biến lịch: phiên đáo hạn mỗi 21 phiên, phiên kế tiếp là phiên chuyển hợp đồng (J = 1, lợi suất khuyết), kỳ nghỉ
dài ngẫu nhiên. Tiêu chí: độ chệch, RMSE, độ phủ khoảng 95% (Wald cho MLE, khoảng tin cậy Bayes cho VB và NUTS),
tỷ lệ bác bỏ sai H0: theta = 0 (E2) và H0: gamma = 0 (E4).
SBC: psi_0 rút từ tiên nghiệm Bảng 5, sinh dữ liệu, suy diễn, ghi thứ hạng của psi_0 trong các mẫu hậu nghiệm;
thứ hạng phân phối đều khi suy diễn đúng.
TỔ CHỨC TÍNH TOÁN: dữ liệu của mỗi lần lặp được sinh lại tất định từ hạt giống riêng của (phân phối, kịch bản, b),
nên các chuỗi NUTS và các lần khởi tạo VB của cùng một lần lặp, dù chạy ở máy nào, đều dùng cùng một bộ dữ liệu.

THƯ VIỆN: numpy, jax (qua uoc_luong), scipy.stats (kiểm định chi bình phương cho thứ hạng).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy import stats

from . import uoc_luong as U
from .mo_hinh_jax import DacTa

CHUNG = {"mu": 0.05, "omega": 0.04, "kappa": 0.15, "beta": 0.95}
KICH_BAN = {"E1": (-0.10, (0.05, 0.05), 8.0, -0.1), "E2": (0.0, (0.05, 0.05), 8.0, -0.1),
            "E3": (-0.10, (0.05, 0.05), 4.0, -0.1), "E4": (-0.10, (0.0, 0.0), 8.0, -0.1)}
BIEN = ("X1", "X2")
SO_MAU_LUU = 250                  # số mẫu hậu nghiệm lưu cho mỗi đơn vị Bayes (float32, thưa đều)


def tham_so_that(kb: str, pp: str) -> np.ndarray:
    th, gam, nu, lam = KICH_BAN[kb]
    p = [CHUNG["mu"], CHUNG["omega"], CHUNG["kappa"], th, CHUNG["beta"], *gam, 0.0, 0.0, 0.0]
    return np.array(p + {"chuan": [], "student": [nu], "skewt": [nu, lam], "ged": [1.5]}[pp])


def _bien_va_lich(n: int, rng):
    X = np.zeros((n, 2))
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + math.sqrt(0.75) * rng.normal(size=2)
    D = np.zeros((n, 3))
    D[20::21, 0] = 1
    D[21::21, 1] = 1
    D[:, 2] = rng.random(n) < 0.006
    Z = np.zeros((n, 5))
    Z[1:, :2] = X[:-1]
    Z[:, 2:] = D
    return Z, D[:, 1] > 0


def sinh(spec: DacTa, psi: np.ndarray, T: int, khoi_dong: int, rng):
    """Dữ liệu mô phỏng ở dạng (r, J, Z, mặt nạ, lh0) cho hàm hợp lý."""
    n = T + khoi_dong
    Z, J = _bien_va_lich(n, rng)
    lh_bd = float(psi[1] / (1 - psi[4]))
    r, _ = U.sinh_tu_mo_hinh(spec, psi, Z, J, lh_bd, rng)
    r, J, Z = r[khoi_dong:], J[khoi_dong:], Z[khoi_dong:]
    lh0 = math.log(float(np.var(r[~J][:20], ddof=1)))
    return (jnp.asarray(r), jnp.asarray(J), jnp.asarray(np.ascontiguousarray(Z)), jnp.asarray(~J), jnp.asarray(lh0))


def rut_tien_nghiem(pp: str, rng) -> np.ndarray:
    """psi rút từ tiên nghiệm Bảng 5 (Phụ lục B cho GED)."""
    p = list(rng.normal(0, 0.5, 4)) + [rng.beta(20, 1.5)] + list(rng.normal(0, 0.5, 2)) + list(rng.normal(0, 1, 3))
    if pp in ("student", "skewt"):
        p.append(2 + rng.gamma(2.0, 10.0))
    if pp == "skewt":
        p.append(2 * rng.beta(2, 2) - 1)
    if pp == "ged":
        p.append(rng.gamma(4.0, 0.5))
    return np.array(p)


def du_lieu_lan(kb: str, pp: str, cfg: dict, seed: int):
    """(spec, psi_0, dữ liệu) của một lần lặp mô phỏng khôi phục; tất định theo seed."""
    spec = DacTa(pp, BIEN)
    psi0 = tham_so_that(kb, pp)
    dl = sinh(spec, psi0, cfg["mo_phong"]["T"], cfg["mo_phong"]["khoi_dong"], np.random.default_rng(seed))
    return spec, psi0, dl


def du_lieu_sbc(pp: str, cfg: dict, seed: int):
    rng = np.random.default_rng(seed)
    spec = DacTa(pp, BIEN)
    psi0 = rut_tien_nghiem(pp, rng)
    return spec, psi0, sinh(spec, psi0, cfg["mo_phong"]["T"], cfg["mo_phong"]["khoi_dong"], rng)


def _mle(spec, dl, cfg, seed):
    return U.mle(spec, dl, cfg["mle"]["so_diem_khoi_tao"], cfg["mle"]["do_tan_khoi_tao"], np.random.default_rng(seed))


# ----------------------------------------------------------------------------------------
# Đơn vị tính toán
# ----------------------------------------------------------------------------------------
def don_vi_mle(spec, psi0, dl, cfg, seed) -> dict:
    """MLE, sai số chuẩn Hessian và bánh kẹp, LR cho gamma = 0 của một lần lặp."""
    m = _mle(spec, dl, cfg, seed)
    se = U.sai_so_chuan(spec, m["psi"], dl)
    long = _mle(DacTa(spec.pp, ()), U.bo_bien(dl, spec.nx), cfg, seed + 1)
    return {"that": psi0, "ten": spec.ten(), "uoc": m["psi"], "se_h": se["se_hessian"], "se_s": se["se_banh_kep"],
            "LR_gamma": 2 * (m["ll"] - long["ll"]), "so_gan_tot": m["so_gan_tot"]}


def don_vi_bayes(spec, psi0, dl, cfg, pt: str, seed_khoi_tao: int, seed: int, chi_so: int, thu_muc) -> dict:
    """Một chuỗi NUTS (pt = 'nuts', chi_so = số chuỗi) hoặc một lần khởi tạo VB (chi_so = số lần) của một lần lặp.
    Khởi tạo tại nghiệm MLE của chính dữ liệu đó (cộng nhiễu nhỏ khác nhau giữa các chuỗi/lần)."""
    b = cfg["bayes"]
    m = _mle(spec, dl, cfg, seed_khoi_tao)
    rng = np.random.default_rng(seed)
    if pt == "nuts":
        r = U.nuts_chuoi(spec, dl, m["xi"] + rng.normal(0, 0.05, spec.d), b["nuts_khoi_dong"], b["nuts_so_mau"],
                         b["nuts_dich_chap_nhan"], b["nuts_do_sau_cay"], seed, thu_muc=thu_muc,
                         vong_moi_doan=b["nuts_vong_moi_doan"])
        chan_doan = {"phan_ky": r["phan_ky"], "buoc_tb": r["buoc_tb"]}
    else:
        r = U.vb(spec, dl, m["xi"] + rng.normal(0, 0.02 * chi_so, spec.d), pt, b["vb_so_buoc"], b["vb_toc_do_hoc"],
                 b["so_mau_hau_nghiem"], seed, thu_muc=thu_muc, buoc_moi_doan=b["vb_buoc_moi_doan"])
        chan_doan = {"elbo": r["elbo"], "k_hat": r["k_hat"]}
    M = r["psi_mau"]
    idx = np.linspace(0, len(M) - 1, min(SO_MAU_LUU, len(M))).astype(int)
    return {"that": psi0, "ten": spec.ten(), "mau": M[idx].astype(np.float32), "giay": r["giay"], **chan_doan}


def gop_bayes(cac_don_vi: list[dict], pt: str, cfg: dict) -> dict:
    """Gộp các chuỗi (NUTS) hoặc chọn lần khởi tạo có ELBO lớn nhất (VB) của một lần lặp."""
    b = cfg["bayes"]
    if pt == "nuts":
        ch = [{"psi_mau": u["mau"].astype(float), "phan_ky": u["phan_ky"]} for u in cac_don_vi]
        cd = U.chan_doan_nuts(ch, b["rhat_toi_da"], b["ess_toi_thieu"]) if len(ch) > 1 else {}
        M = np.concatenate([c["psi_mau"] for c in ch])
        cd = {k: cd.get(k) for k in ("rhat_max", "ess_min", "phan_ky", "dat")}
    else:
        tot = max(cac_don_vi, key=lambda u: u["elbo"])
        M = tot["mau"].astype(float)
        cd = {"k_hat": tot["k_hat"], "elbo_cac_lan": [u["elbo"] for u in cac_don_vi],
              "dat": bool(tot["k_hat"] < b["k_hat_toi_da"])}
    u0 = cac_don_vi[0]
    return {"that": u0["that"], "ten": u0["ten"], "mau": M, "uoc": M.mean(0), "duoi": np.quantile(M, 0.025, 0),
            "tren": np.quantile(M, 0.975, 0), "chan_doan": cd}


# ----------------------------------------------------------------------------------------
# Tổng hợp
# ----------------------------------------------------------------------------------------
def tong_hop(kq: list[dict], pp_suy_dien: str) -> dict:
    ten, T0 = kq[0]["ten"], np.array([x["that"] for x in kq])
    U_ = np.array([x["uoc"] for x in kq])
    out = {"B": len(kq), "tham_so": {}}
    if pp_suy_dien == "mle":
        SH, SS = np.array([x["se_h"] for x in kq]), np.array([x["se_s"] for x in kq])
        lo_s, hi_s = U_ - 1.959964 * SS, U_ + 1.959964 * SS
        phu_h = np.mean(np.abs(U_ - T0) <= 1.959964 * SH, 0)
    else:
        lo_s, hi_s = np.array([x["duoi"] for x in kq]), np.array([x["tren"] for x in kq])
        phu_h = None
        out["chan_doan_dat"] = float(np.mean([bool(x["chan_doan"].get("dat")) for x in kq]))
    phu = np.mean((T0 >= lo_s) & (T0 <= hi_s), 0)
    for i, t in enumerate(ten):
        out["tham_so"][t] = {"that": float(T0[0, i]), "chech": float((U_[:, i] - T0[:, i]).mean()),
                             "rmse": float(np.sqrt(((U_[:, i] - T0[:, i]) ** 2).mean())), "do_phu": float(phu[i]),
                             **({"do_phu_hessian": float(phu_h[i])} if phu_h is not None else {})}
    ith = ten.index("theta")
    out["bac_bo_theta0"] = float(np.mean((lo_s[:, ith] > 0) | (hi_s[:, ith] < 0)))
    ig = [ten.index("gamma_X1"), ten.index("gamma_X2")]
    if pp_suy_dien == "mle":
        out["bac_bo_gamma0"] = float(np.mean(np.array([x["LR_gamma"] for x in kq]) > stats.chi2.ppf(0.95, 2)))
    else:   # khoảng tin cậy 95% của ít nhất một gamma_j không chứa 0
        out["bac_bo_gamma0"] = float(np.mean(((lo_s[:, ig] > 0) | (hi_s[:, ig] < 0)).any(1)))
    return out


def hang_sbc(lan: dict, L: int = 100) -> dict:
    M = lan["mau"]
    M = M[np.linspace(0, len(M) - 1, min(L, len(M))).astype(int)]
    return {"hang": (M < lan["that"]).sum(0), "L": len(M), "ten": lan["ten"], "chan_doan": lan["chan_doan"]}


def tong_hop_sbc(kq: list[dict], so_ngan: int = 10) -> dict:
    ten, L = kq[0]["ten"], kq[0]["L"]
    H = np.array([x["hang"] for x in kq])
    out = {}
    for i, t in enumerate(ten):
        dem = np.histogram(H[:, i], bins=so_ngan, range=(0, L + 1))[0]
        out[t] = {"dem": dem.tolist(), "chi2_p": float(stats.chisquare(dem).pvalue)}
    return {"so_lan": len(kq), "L": L, "tham_so": out}
