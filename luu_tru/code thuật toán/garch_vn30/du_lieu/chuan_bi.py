"""Chuẩn bị dữ liệu cho mô hình: biến ngoại sinh, chuẩn hóa, chia tập theo thời gian.

BIẾN NGOẠI SINH (dịch trễ một phiên -- chỉ dùng thông tin đã biết trước ngày t)
    log_volume_lag1 = log(1 + VOL_{t-1})         khối lượng giao dịch phiên trước
    log_rv_lag1     = log(RV_{t-1} + 1e-6)       biến động thực tế phiên trước
    (1e-6 tránh log(0) nếu có ngày RV = 0.)

CHUẨN HÓA KHÔNG RÒ RỈ THÔNG TIN
    Mỗi biến được chuẩn hóa x~ = (x - trung bình) / độ lệch chuẩn, trong đó trung
    bình và độ lệch chuẩn tính CHỈ trên tập huấn luyện. Nếu dùng thống kê của cả
    mẫu, thông tin của tập kiểm định sẽ rò rỉ vào mô hình và làm đánh giá ngoài
    mẫu lạc quan hơn thực tế.

CHIA TẬP
    80% phiên đầu làm tập huấn luyện, 20% phiên sau làm tập kiểm định, KHÔNG xáo
    trộn (chuỗi thời gian phải giữ thứ tự). Với dữ liệu hiện có: 2.170 quan sát,
    1.735 huấn luyện, 435 kiểm định.

THƯ VIỆN
    pandas (bảng, shift, mean/std theo mặt nạ), numpy (log), torch (chuyển sang tensor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from garch_vn30.du_lieu.doc_du_lieu import load_vn30f1m_daily

COT_X = ["log_volume_lag1", "log_rv_lag1"]   # tên và thứ tự các cột của ma trận X


def build_exogenous_matrix(daily: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Tạo hai biến ngoại sinh đã dịch trễ một phiên (shift(1)), kèm cột `date`."""
    out = pd.DataFrame({"date": daily["date"]})
    out["log_volume_lag1"] = np.log1p(daily["volume"]).shift(1)
    out["log_rv_lag1"] = np.log(daily["RV"] + eps).shift(1)
    return out


def standardize_with_train_stats(X: pd.DataFrame, train_mask: pd.Series
                                 ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Chuẩn hóa (x - mean_train) / sd_train cho mọi cột số (trừ `date`).

    Tham số:
        X         : bảng biến ngoại sinh.
        train_mask: Series True/False, True ở các dòng thuộc tập huấn luyện.
    Trả về: (bảng đã chuẩn hóa, trung bình, độ lệch chuẩn) -- hai giá trị sau lưu
    lại để áp dụng đúng cách chuẩn hóa này cho dữ liệu mới.
    """
    cot = [c for c in X.columns if c != "date" and pd.api.types.is_numeric_dtype(X[c])]
    mean = X.loc[train_mask, cot].mean()
    std = X.loc[train_mask, cot].std().replace(0.0, 1.0)   # tránh chia cho 0
    out = X.copy()
    out[cot] = (X[cot] - mean) / std
    return out, mean, std


def chuan_bi_du_lieu(csv_path: str, train_frac: float = 0.8) -> dict:
    """Toàn bộ khâu chuẩn bị dữ liệu: đọc tệp phút, tạo biến, chuẩn hóa, chia tập.

    Mặt nạ huấn luyện đặt trên toàn bộ phiên gốc TRƯỚC khi lọc các dòng thiếu
    (ngày đầu không có lợi suất, không có biến trễ).

    Trả về dict (được lưu thành results/00_data.pt):
        r_all    : tensor lợi suất (T,)
        X_all    : tensor biến ngoại sinh đã chuẩn hóa (T, 2)
        rv_all   : tensor biến động thực tế (T,)
        dates_all: ngày tương ứng (pandas Series)
        n_train  : số quan sát huấn luyện (T - n_train là số quan sát kiểm định)
        quality_report, exog_mean, exog_std: báo cáo chất lượng và tham số chuẩn hóa.
    """
    daily, bao_cao = load_vn30f1m_daily(csv_path)
    X_raw = build_exogenous_matrix(daily)
    n = len(daily)
    train_mask = pd.Series(False, index=X_raw.index)
    train_mask.iloc[: int(n * train_frac)] = True
    X_std, x_mean, x_std = standardize_with_train_stats(X_raw, train_mask)

    hop_le = (daily["log_return"].notna() & X_std["log_volume_lag1"].notna()
              & X_std["log_rv_lag1"].notna() & daily["RV"].notna())
    return {
        "r_all": torch.tensor(daily.loc[hop_le, "log_return"].to_numpy()),
        "X_all": torch.tensor(X_std.loc[hop_le, COT_X].to_numpy()),
        "rv_all": torch.tensor(daily.loc[hop_le, "RV"].to_numpy()),
        "dates_all": daily.loc[hop_le, "date"].reset_index(drop=True),
        "n_train": int(train_mask.loc[hop_le].sum()),
        "quality_report": bao_cao,
        "exog_mean": x_mean,
        "exog_std": x_std,
    }
