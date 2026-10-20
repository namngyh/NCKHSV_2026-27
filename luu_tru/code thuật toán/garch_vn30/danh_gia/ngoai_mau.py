"""Đánh giá dự báo ngoài mẫu cho GARCH/EGARCH một trạng thái.

THIẾT KẾ
    Tham số được ước lượng MỘT LẦN trên tập huấn luyện. Vòng lặp phương sai sau
    đó chạy tiếp qua tập kiểm định với tham số cố định, điều kiện đầu là phương
    sai mẫu của tập huấn luyện. Tại mỗi phiên t của tập kiểm định, h_hat_t chỉ
    dùng thông tin đến t-1, nên đây là dự báo một bước ngoài mẫu hợp lệ (không có
    thông tin tương lai rò rỉ vào dự báo). Tham số không được ước lượng lại theo ngày.

HAI CÁCH DỰ BÁO
    1. Một bộ tham số (MLE): h_hat_t và mật độ f(r_t | theta_hat).
    2. Hậu nghiệm (VB/MCMC), S bộ tham số theta^(s):
       - phương sai dự báo  h_bar_t = (1/S) sum_s h_t^(s);
       - mật độ dự báo HỖN HỢP  f_t(r) = (1/S) sum_s f(r | theta^(s)),
         tính trên thang log bằng logsumexp để không tràn số:
             log f_t(r) = logsumexp_s log f(r | theta^(s)) - log S.
       Qua mật độ hỗn hợp, bất định về tham số đi vào dự báo.

CHỈ TIÊU TRẢ VỀ
    mae, rmse, qlike (phương sai so với RV), nls (mật độ), h_hat, n_test, và
    với VaR 1% và 5%: kupiec_<muc>, christoffersen_<muc>, n_violations_<muc>.
"""
from __future__ import annotations

import torch

from garch_vn30.danh_gia.ham_mat_mat import mae_vol, negative_log_score, qlike, rmse_vol
from garch_vn30.danh_gia.kiem_dinh_var import christoffersen_test, kupiec_test, phan_vi_z
from garch_vn30.mo_hinh.danh_muc import forward_cua
from garch_vn30.mo_hinh.garch import log_mat_do

MUC_VAR = [(0.01, "1pct"), (0.05, "5pct")]   # (mức a, nhãn trong dict kết quả)


def duong_h_va_logf(params, family: str, r_all: torch.Tensor, X_all: torch.Tensor,
                    n_train: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Trả về (h_hat, log f) trên TẬP KIỂM ĐỊNH cho một bộ tham số.

    Chạy vòng lặp phương sai qua TOÀN BỘ chuỗi (huấn luyện + kiểm định) rồi cắt
    lấy phần từ n_train trở đi. torch.no_grad(): chỉ tính giá trị, không cần đạo hàm.
    """
    with torch.no_grad():
        h0 = torch.var(r_all[:n_train]).detach()
        h_path = forward_cua(family)(params, r_all, X_all, h0=h0)
    h_hat, r_test = h_path[n_train:], r_all[n_train:]
    return h_hat, log_mat_do(r_test, params.mu, h_hat, params.nu)


def _kiem_dinh_var(ket_qua: dict, r_test: torch.Tensor, var_theo_muc: dict) -> None:
    """Thêm số vi phạm và kiểm định Kupiec/Christoffersen cho từng mức VaR vào `ket_qua`."""
    for alpha, nhan in MUC_VAR:
        hits = (r_test.numpy() < -var_theo_muc[alpha].numpy()).astype(float)   # I_t
        ket_qua[f"kupiec_{nhan}"] = kupiec_test(hits, alpha)
        ket_qua[f"christoffersen_{nhan}"] = christoffersen_test(hits, alpha)
        ket_qua[f"n_violations_{nhan}"] = int(hits.sum())


def danh_gia_mot_diem(params, family: str, dist: str, r_all: torch.Tensor,
                      X_all: torch.Tensor, rv_all: torch.Tensor, n_train: int) -> dict:
    """Đánh giá ngoài mẫu cho MỘT bộ tham số (dự báo "cắm" tham số, vd. MLE).

    Tham số:
        params          : GARCHParams hoặc EGARCHParams.
        family, dist    : họ mô hình và phân phối sai số.
        r_all, X_all    : toàn bộ chuỗi lợi suất và ngoại sinh (huấn luyện + kiểm định).
        rv_all          : biến động thực tế RV_t (cùng đơn vị với h).
        n_train         : số quan sát huấn luyện; phần còn lại là tập kiểm định.
    Trả về: dict chỉ tiêu (xem đầu tệp).
    """
    h_hat, log_f = duong_h_va_logf(params, family, r_all, X_all, n_train)
    rv_test, r_test = rv_all[n_train:], r_all[n_train:]
    ket_qua = {
        "mae": float(mae_vol(rv_test, h_hat)),
        "rmse": float(rmse_vol(rv_test, h_hat)),
        "qlike": float(qlike(rv_test, h_hat)),
        "nls": float(negative_log_score(log_f)),
        "h_hat": h_hat,
        "n_test": int(rv_test.shape[0]),
    }
    nu = None if dist == "normal" else float(params.nu)
    # VaR_t(a) = -(mu + sqrt(h_t) q_a)
    var = {a: -(float(params.mu) + torch.sqrt(h_hat) * phan_vi_z(a, dist, nu))
           for a, _ in MUC_VAR}
    _kiem_dinh_var(ket_qua, r_test, var)
    return ket_qua


def danh_gia_hau_nghiem(draws: list, family: str, dist: str, r_all: torch.Tensor,
                        X_all: torch.Tensor, rv_all: torch.Tensor, n_train: int) -> dict:
    """Đánh giá ngoài mẫu khi có S bộ tham số rút từ hậu nghiệm (VB hoặc MCMC).

    `draws` là danh sách bộ tham số (xem suy_dien/bayes.rut_bo_tham_so). VaR lấy
    bằng trung bình phân vị của các thành phần (xấp xỉ phân vị của hỗn hợp).
    Với đúng một bộ tham số, kết quả trùng danh_gia_mot_diem (có kiểm thử).
    """
    rv_test, r_test = rv_all[n_train:], r_all[n_train:]
    h_ds, logf_ds = [], []
    for params in draws:
        h_hat, log_f = duong_h_va_logf(params, family, r_all, X_all, n_train)
        h_ds.append(h_hat)
        logf_ds.append(log_f)
    h_draws = torch.stack(h_ds)            # (S, n_test)
    logf_draws = torch.stack(logf_ds)      # (S, n_test)

    h_mean = h_draws.mean(dim=0)
    log_mix = torch.logsumexp(logf_draws, dim=0) - torch.log(torch.tensor(float(len(draws))))
    ket_qua = {
        "mae": float(mae_vol(rv_test, h_mean)),
        "rmse": float(rmse_vol(rv_test, h_mean)),
        "qlike": float(qlike(rv_test, h_mean)),
        "nls": float(negative_log_score(log_mix)),
        "h_hat": h_mean,
        "n_test": int(rv_test.shape[0]),
        "n_draws_used": int(h_draws.shape[0]),
    }

    dt = r_all.dtype
    mu_draws = torch.stack([p.mu.reshape(()) for p in draws])
    var = {}
    for a, _ in MUC_VAR:
        if dist == "normal":
            q = torch.full((len(draws),), phan_vi_z(a, "normal", None), dtype=dt)
        else:
            q = torch.tensor([phan_vi_z(a, dist, float(p.nu)) for p in draws], dtype=dt)
        var[a] = -(mu_draws.mean() + (torch.sqrt(h_draws) * q[:, None]).mean(dim=0))
    _kiem_dinh_var(ket_qua, r_test, var)
    return ket_qua
