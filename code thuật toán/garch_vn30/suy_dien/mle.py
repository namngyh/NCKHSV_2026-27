"""Ước lượng hợp lý cực đại (MLE) cho GARCH/EGARCH một trạng thái.

LÝ THUYẾT
    MLE chọn tham số làm cho dữ liệu quan sát "hợp lý nhất":
        theta_hat = argmax_theta  l(theta),    l = log-hợp lý (mo_hinh/garch.py, egarch.py).
    Tương đương: cực tiểu hàm mất mát L(theta) = -l(theta).

    1) THAM SỐ HÓA KHÔNG RÀNG BUỘC. Thuật toán tối ưu làm việc trên R^d, nhưng
       tham số có miền giới hạn (omega > 0, nu > 2, ...). Vì vậy mỗi tham số được
       viết qua một biến tự do eta ∈ R:
           GARCH : omega = exp(log_omega), alpha = softplus(eta_alpha), beta = softplus(eta_beta)
           EGARCH: omega, alpha, theta tự do;   beta = sigmoid(eta_beta) ∈ (0, 1)
           cả hai: mu, gamma tự do;   nu = 2 + softplus(eta_nu)
       với softplus(x) = log(1 + e^x) > 0 và sigmoid(x) = 1 / (1 + e^-x) ∈ (0, 1).

    2) NHIỀU ĐIỂM KHỞI TẠO. l có thể có nhiều cực đại địa phương, nên chạy từ
       n_starts điểm xuất phát ngẫu nhiên và giữ nghiệm có l lớn nhất.

    3) HAI GIAI ĐOẠN TỐI ƯU cho mỗi điểm khởi tạo:
       - Adam (thuật toán gradient thích nghi, Kingma & Ba 2015): ổn định khi còn
         xa nghiệm; chuẩn gradient được cắt ở 50 để một bước lớn không đẩy tham
         số ra vùng số học bất ổn.
       - L-BFGS (tựa Newton, dùng đường cong bậc hai xấp xỉ từ lịch sử gradient)
         với tìm kiếm đường Wolfe mạnh: hội tụ nhanh và chính xác khi đã gần nghiệm.

    4) SAI SỐ CHUẨN. Theo lý thuyết tiệm cận, Cov(eta_hat) ≈ H^{-1} với H là
       Hessian (ma trận đạo hàm bậc hai) của -l tại nghiệm. Phương pháp delta đưa
       về thang tự nhiên: Cov(theta_hat) ≈ J H^{-1} J^T, J = d theta / d eta.
       Khoảng tin cậy Wald 95%: theta_hat ± 1,96 SE.

    5) TIÊU CHÍ THÔNG TIN: AIC = 2d - 2l, BIC = d log(n) - 2l (nhỏ hơn là tốt hơn).

THƯ VIỆN
    torch.optim.Adam, torch.optim.LBFGS  : hai bộ tối ưu nói trên.
    loss.backward()                      : vi phân tự động, tính gradient của L theo eta.
    torch.nn.utils.clip_grad_norm_       : cắt chuẩn gradient.
    torch.autograd.functional.hessian    : Hessian chính xác (không dùng sai phân hữu hạn).
    torch.autograd.functional.jacobian   : ma trận Jacobian J cho phương pháp delta.
    torch.linalg.inv                     : nghịch đảo ma trận.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from garch_vn30.mo_hinh.danh_muc import loglik_cua
from garch_vn30.mo_hinh.egarch import EGARCHParams
from garch_vn30.mo_hinh.garch import GARCHParams

_softplus = torch.nn.functional.softplus

# Thứ tự các biến không ràng buộc khi "phẳng hóa" thành một vector để tính
# Hessian, và tên tham số tự nhiên tương ứng theo đúng thứ tự đó.
_BIEN_RAW = {
    "garch": ["mu", "log_omega", "eta_alpha", "eta_beta", "gamma"],
    "egarch": ["mu", "omega", "alpha", "theta", "eta_beta", "gamma"],
}
_TEN_TU_NHIEN = {
    "garch": ["mu", "omega", "alpha", "beta", "gamma"],
    "egarch": ["mu", "omega", "alpha", "theta", "beta", "gamma"],
}


@dataclass
class LanChay:
    """Chẩn đoán của một điểm khởi tạo."""
    seed: int            # hạt giống sinh điểm khởi tạo
    loglik: float        # log-hợp lý tại điểm dừng
    converged: bool      # log-hợp lý hữu hạn và ||gradient|| < 1 tại điểm dừng
    grad_norm: float     # chuẩn gradient tại điểm dừng


@dataclass
class MLEResult:
    """Kết quả MLE: nghiệm tốt nhất và các chỉ tiêu đi kèm."""
    params: GARCHParams | EGARCHParams   # tham số ước lượng (thang tự nhiên)
    loglik: float                         # log-hợp lý tại nghiệm
    aic: float
    bic: float
    n_params: int                         # số tham số tự do d
    runs: list[LanChay] = field(default_factory=list)   # chẩn đoán từng điểm khởi tạo


# ------------------------------------------------------------ biến đổi tham số
def unpack(raw: dict, dist: str, family: str):
    """Biến không ràng buộc (dict tên -> tensor) -> bộ tham số trên miền hợp lệ.

    Xem mục 1 ở đầu tệp. Mọi phép biến đổi đều khả vi, nên gradient theo eta
    được PyTorch tính tự động qua các phép biến đổi này.
    """
    nu = 2.0 + _softplus(raw["eta_nu"]) if dist == "student_t" else None
    if family == "garch":
        return GARCHParams(
            mu=raw["mu"],
            omega=torch.exp(raw["log_omega"]),
            alpha=_softplus(raw["eta_alpha"]),
            beta=_softplus(raw["eta_beta"]),
            gamma=raw["gamma"],
            nu=nu,
        )
    if family == "egarch":
        # beta ∈ (0,1) chứ không phải (-1,1): tiên nghiệm Bayes của beta là
        # Beta(a,a) trên (0,1), nên MLE cũng tìm trên (0,1) để ba phương pháp
        # làm việc trên CÙNG một không gian tham số.
        return EGARCHParams(
            mu=raw["mu"],
            omega=raw["omega"],
            alpha=raw["alpha"],
            theta=raw["theta"],
            beta=torch.sigmoid(raw["eta_beta"]),
            gamma=raw["gamma"],
            nu=nu,
        )
    raise ValueError(f"family khong hop le: {family!r}")


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Nghịch đảo softplus: x = log(e^y - 1); torch.expm1(y) = e^y - 1 chính xác khi y nhỏ."""
    return torch.log(torch.expm1(y))


def pack(params, dist: str, family: str) -> dict:
    """Chiều ngược của unpack: bộ tham số tự nhiên -> biến không ràng buộc.

    Cần khi tính Hessian tại một nghiệm đã lưu (Hessian phải lấy trên không gian
    mà việc tối ưu thực sự diễn ra).
    """
    if family == "garch":
        raw = {"mu": params.mu, "log_omega": torch.log(params.omega),
               "eta_alpha": _inv_softplus(params.alpha),
               "eta_beta": _inv_softplus(params.beta), "gamma": params.gamma}
    else:
        raw = {"mu": params.mu, "omega": params.omega, "alpha": params.alpha,
               "theta": params.theta,
               "eta_beta": torch.log(params.beta / (1.0 - params.beta)),   # logit = nghịch đảo sigmoid
               "gamma": params.gamma}
    if dist == "student_t":
        raw["eta_nu"] = _inv_softplus(params.nu - 2.0)
    return raw


def _khoi_tao(dist: str, family: str, k: int, h0: torch.Tensor) -> dict:
    """Điểm khởi tạo ngẫu nhiên quanh một cấu hình hợp lý (đặt seed trước khi gọi).

    GARCH : omega ≈ 0,1 h0, alpha ≈ softplus(-1,5) ≈ 0,20, beta ≈ softplus(1) ≈ 1,31.
    EGARCH: beta ≈ 0,9 và omega ≈ (1 - beta) log h0 để mức nền của log h khớp dữ
            liệu; theta quanh 0 để KHÔNG giả định trước dấu đòn bẩy.
    Thứ tự các lệnh torch.randn cố định để cùng seed luôn cho cùng điểm khởi tạo.
    """
    if family == "garch":
        raw = {
            "mu": torch.zeros(()),
            "log_omega": torch.log(h0 * 0.1) + 0.1 * torch.randn(()),
            "eta_alpha": torch.tensor(-1.5) + 0.2 * torch.randn(()),
            "eta_beta": torch.tensor(1.0) + 0.2 * torch.randn(()),
            "gamma": 0.01 * torch.randn(k),
        }
    else:
        beta0 = 0.9
        raw = {
            "mu": torch.zeros(()),
            "omega": (1.0 - beta0) * torch.log(h0) + 0.05 * torch.randn(()),
            "alpha": torch.tensor(0.10) + 0.02 * torch.randn(()),
            "theta": torch.tensor(0.0) + 0.02 * torch.randn(()),
            "eta_beta": torch.log(torch.tensor(beta0 / (1.0 - beta0))) + 0.1 * torch.randn(()),
            "gamma": 0.01 * torch.randn(k),
        }
    if dist == "student_t":
        raw["eta_nu"] = torch.tensor(1.5) + 0.2 * torch.randn(())   # nu ≈ 2 + softplus(1,5) ≈ 3,7
    return raw


# ------------------------------------------------------------------ tối ưu
def _mot_diem_khoi_tao(seed, r, X, dist, family, h0, n_adam, n_lbfgs, lr_adam, verbose):
    """Tối ưu từ MỘT điểm khởi tạo: Adam rồi L-BFGS. Trả về (params, loglik, LanChay)."""
    torch.manual_seed(seed)
    loglik = loglik_cua(family)
    raw = _khoi_tao(dist, family, X.shape[1], h0)
    # requires_grad_(True): yêu cầu PyTorch theo dõi để tính gradient theo các biến này.
    vars_ = {ten: v.clone().requires_grad_(True) for ten, v in raw.items()}
    ds_bien = list(vars_.values())

    def nll():
        """Hàm mất mát L = -l (negative log-likelihood)."""
        return -loglik(unpack(vars_, dist, family), r, X, h0=h0)

    # Giai đoạn 1 -- Adam.
    opt = torch.optim.Adam(ds_bien, lr=lr_adam)
    for it in range(n_adam):
        opt.zero_grad()                  # xóa gradient của bước trước
        loss = nll()
        if torch.isnan(loss):            # tham số rơi vào vùng không hợp lệ: dừng Adam
            break
        loss.backward()                  # vi phân tự động: tính dL/d(eta)
        torch.nn.utils.clip_grad_norm_(ds_bien, 50.0)
        opt.step()                       # cập nhật eta theo quy tắc Adam
        if verbose and it % 200 == 0:
            print(f"  [seed={seed}] adam {it}: nll={loss.item():.4f}")

    # Giai đoạn 2 -- L-BFGS với tìm kiếm đường Wolfe mạnh. LBFGS của PyTorch cần
    # một hàm "closure" tính lại loss và gradient mỗi khi nó thử một điểm mới.
    lbfgs = torch.optim.LBFGS(ds_bien, lr=0.5, max_iter=n_lbfgs, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = nll()
        if torch.isnan(loss):
            return loss
        loss.backward()
        return loss

    truoc_lbfgs = {ten: v.detach().clone() for ten, v in vars_.items()}
    try:
        lbfgs.step(closure)
    except (IndexError, RuntimeError) as exc:
        # L-BFGS của torch có thể vỡ khi một bước tìm kiếm đường cho NaN: quay
        # về nghiệm của giai đoạn Adam thay vì làm hỏng cả quá trình.
        print(f"  [seed={seed}] L-BFGS that bai ({type(exc).__name__}); giu nghiem Adam.")
        with torch.no_grad():
            for ten, v in vars_.items():
                v.copy_(truoc_lbfgs[ten])
                v.grad = None
        lai = nll()
        if torch.isfinite(lai):
            lai.backward()

    # Chẩn đoán tại điểm dừng: giá trị hàm mục tiêu và chuẩn gradient.
    with torch.no_grad():                # no_grad: chỉ tính giá trị, không theo dõi đạo hàm
        final_loss = nll()
    grad_norm = float(sum((v.grad.norm() ** 2 for v in ds_bien if v.grad is not None),
                          torch.tensor(0.0)).sqrt())
    converged = bool(torch.isfinite(final_loss)) and grad_norm < 1.0
    params = unpack({ten: v.detach() for ten, v in vars_.items()}, dist, family)
    ll = float(-final_loss.item())
    return params, ll, LanChay(seed=seed, loglik=ll, converged=converged, grad_norm=grad_norm)


def fit_mle(r: torch.Tensor, X: torch.Tensor, dist: str = "student_t", family: str = "garch",
            n_starts: int = 5, base_seed: int = 0, n_adam: int = 400, n_lbfgs: int = 200,
            lr_adam: float = 0.02, h0: torch.Tensor | None = None,
            verbose: bool = False) -> MLEResult:
    """Ước lượng hợp lý cực đại với nhiều điểm khởi tạo.

    Tham số:
        r, X     : chuỗi lợi suất (T,) và ma trận ngoại sinh (T, k); không dùng X thì k = 0.
        dist     : "normal" hoặc "student_t".
        family   : "garch" hoặc "egarch".
        n_starts : số điểm khởi tạo; điểm thứ i dùng seed = base_seed + i.
        n_adam, n_lbfgs, lr_adam: số bước Adam, số vòng L-BFGS, tốc độ học Adam.
        h0       : điều kiện đầu của phương sai (mặc định: phương sai mẫu của r).
        verbose  : in tiến độ Adam.
    Trả về:
        MLEResult với nghiệm có log-hợp lý HỮU HẠN lớn nhất, AIC, BIC, d và chẩn
        đoán từng điểm khởi tạo.
    """
    if h0 is None:
        h0 = torch.var(r).detach()
    best, runs = None, []
    for i in range(n_starts):
        params, ll, lan = _mot_diem_khoi_tao(base_seed + i, r, X, dist, family, h0,
                                             n_adam, n_lbfgs, lr_adam, verbose)
        runs.append(lan)
        if math.isfinite(ll) and (best is None or ll > best[1]):
            best = (params, ll)
    if best is None:
        raise RuntimeError("Tat ca cac diem khoi tao MLE deu that bai (NaN/Inf).")

    params, ll = best
    d = 4 + (1 if family == "egarch" else 0) + X.shape[1] + (1 if dist == "student_t" else 0)
    aic = -2.0 * ll + 2.0 * d
    bic = -2.0 * ll + d * torch.log(torch.tensor(float(r.shape[0]))).item()
    return MLEResult(params=params, loglik=ll, aic=aic, bic=bic, n_params=d, runs=runs)


# ------------------------------------------------------------ sai số chuẩn
def _phang(raw: dict, dist: str, family: str) -> torch.Tensor:
    """Ghép các biến không ràng buộc thành MỘT vector (để tính Hessian)."""
    ten = _BIEN_RAW[family] + (["eta_nu"] if dist == "student_t" else [])
    return torch.cat([raw[t].reshape(-1) for t in ten])


def _dung_lai(vec: torch.Tensor, k: int, dist: str, family: str) -> dict:
    """Chiều ngược của _phang: tách vector trở lại dict theo tên."""
    ten = _BIEN_RAW[family] + (["eta_nu"] if dist == "student_t" else [])
    raw, pos = {}, 0
    for t in ten:
        n = k if t == "gamma" else 1
        raw[t] = vec[pos] if n == 1 else vec[pos:pos + n]
        pos += n
    return raw


def standard_errors(params, r: torch.Tensor, X: torch.Tensor, dist: str = "student_t",
                    family: str = "garch", h0: torch.Tensor | None = None
                    ) -> dict[str, torch.Tensor] | None:
    """Sai số chuẩn của tham số trên thang TỰ NHIÊN, bằng Hessian quan sát + phương pháp delta.

        H  = d^2(-l) / d eta^2      tại nghiệm (Hessian quan sát)
        J  = d theta / d eta        (Jacobian của phép biến đổi unpack)
        Cov(theta_hat) ≈ J H^{-1} J^T,   SE_j = căn bậc hai phần tử đường chéo thứ j

    Tham số: params (nghiệm MLE), r, X, dist, family như fit_mle.
    Trả về: dict {tên tham số: SE} (gamma là vector), hoặc None khi Hessian không
    khả nghịch hay cho phương sai âm -- nghiệm khi đó không phải cực đại trong
    (thường do nằm sát biên), tiệm cận chuẩn không áp dụng, khoảng Wald không hợp lệ.
    """
    k = X.shape[1]
    if h0 is None:
        h0 = r.var()
    loglik = loglik_cua(family)
    vec = _phang(pack(params, dist, family), dist, family).detach().clone()

    def nll(v):
        return -loglik(unpack(_dung_lai(v, k, dist, family), dist, family), r, X, h0=h0)

    hess = torch.autograd.functional.hessian(nll, vec)
    try:
        cov_raw = torch.linalg.inv(hess)
    except RuntimeError:
        return None
    if (torch.diagonal(cov_raw) < 0).any():
        return None

    def sang_tu_nhien(v):
        """eta -> vector tham số tự nhiên, để lấy Jacobian J."""
        p = unpack(_dung_lai(v, k, dist, family), dist, family)
        phan = [p.mu.reshape(-1), p.omega.reshape(-1), p.alpha.reshape(-1)]
        if family == "egarch":
            phan.append(p.theta.reshape(-1))
        phan += [p.beta.reshape(-1), p.gamma.reshape(-1)]
        if dist == "student_t":
            phan.append(p.nu.reshape(-1))
        return torch.cat(phan)

    J = torch.autograd.functional.jacobian(sang_tu_nhien, vec)
    se_phang = torch.sqrt(torch.diagonal(J @ cov_raw @ J.T).clamp(min=0.0))

    ten = _TEN_TU_NHIEN[family] + (["nu"] if dist == "student_t" else [])
    ra, pos = {}, 0
    for t in ten:
        n = k if t == "gamma" else 1
        ra[t] = se_phang[pos:pos + n] if t == "gamma" else se_phang[pos]
        pos += n
    return ra
