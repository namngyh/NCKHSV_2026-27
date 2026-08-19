"""Xuất xác suất chế độ biến động (MS-GARCH-X, bộ lọc tiến) ra CSV cho QP
Analysis (dự án backtest tách rời - xem Desktop/QP Platform) đọc vào làm cột
ngoại sinh point-in-time.

ĐÂY LÀ BẢN PROTOTYPE, KHÔNG PHẢI WALK-FORWARD: tham số MS-GARCH-X được ước
lượng MLE MỘT LẦN trên đoạn train của QP, rồi bộ lọc tiến chạy suốt lịch sử
với tham số cố định đó. README của package này đã ghi: một mô hình MLE 2
trạng thái mất khoảng 2.5 giờ ở cấu hình full (đo thực nghiệm, xem "Lưu ý
hiệu năng" trong README.md), còn MCMC (NUTS) có thể mất hơn một ngày tùy
hình học hậu nghiệm -- refit lặp lại theo từng fold walk-forward là không
khả thi trên CPU với bộ lọc tuần tự này. Xem cảnh báo tương ứng in ra khi
chiến lược qp sử dụng chuỗi này (qp/strategy/library.py::vol_regime_gate).

Hai điểm point-in-time quan trọng:

1. Bộ lọc tiến `xi_filt[t]` CẦN lợi suất đóng cửa phiên t để cập nhật -> chưa
   biết được lúc mở phiên t. File này xuất giá trị CHƯA lag; bên đọc (qp) BẮT
   BUỘC phải lag thêm ít nhất 1 phiên trước khi dùng trong phiên giao dịch
   (qp/data/exogenous.py::attach_daily_signal, mặc định lag_days=1).
2. `h0` (phương sai khởi tạo bộ lọc) được ép bằng phương sai đo trên ĐOẠN
   TRAIN, không phải toàn mẫu - nếu không giá trị khởi tạo tại t=0 sẽ "nhìn"
   được biến động của cả validation/test.

Chạy:
    python scripts/05_export_regime_signal_for_qp.py
    python scripts/05_export_regime_signal_for_qp.py --train-end 2022-12-02 --n-starts 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msgarchx.data.exogenous import build_exogenous_matrix, standardize_with_train_stats
from msgarchx.data.loader import load_vn30f1m_daily
from msgarchx.inference.mle import fit_mle
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.models.ms_garch import (
    expected_duration,
    label_switch_correction,
    ms_garch_forward_filter,
)

# Ranh giới train của QP Analysis (qp/config.py, config/default.yaml:
# splits.mode=fractions, train=0.60 trên nến 15 phút VN30F1M). Đo trực tiếp
# từ dữ liệu hiện tại qua `python scripts/backtest.py --list` phía qp lúc
# ingest 2026-08-09. Hai kho mã KHÔNG chia sẻ code nên không tự suy ra được -
# đổi file ohlc_export.csv (nạp lại dữ liệu) ở BÊN QP thì phải đo lại ranh
# giới này và truyền qua --train-end, nếu không tập fit ở đây sẽ lệch khỏi
# tập train thật của qp.
DEFAULT_TRAIN_END = "2022-12-02"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "csv_path", nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "ohlc_export.csv"),
    )
    ap.add_argument(
        "--train-end", default=DEFAULT_TRAIN_END,
        help="ngày cuối dùng để fit MLE - PHẢI khớp ranh giới train của qp",
    )
    ap.add_argument("--n-starts", type=int, default=4, help="số điểm khởi tao MLE")
    ap.add_argument("--n-adam", type=int, default=300)
    ap.add_argument("--n-lbfgs", type=int, default=150)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "regime_signal_vn30f1m.csv"),
    )
    args = ap.parse_args()

    t_start = time.time()
    daily, report = load_vn30f1m_daily(args.csv_path)
    print(f"Nạp {report['n_trading_days']} phiên, {report['date_min'].date()} -> {report['date_max'].date()}")

    X_raw = build_exogenous_matrix(
        daily, volume_col="volume", rv_col="RV",
        basis_col=None, expiry_dummy_col=None, roll_dummy_col=None,
    )
    train_end = pd.Timestamp(args.train_end)
    train_mask_daily = daily["date"] <= train_end
    if train_mask_daily.sum() < 500:
        raise ValueError(
            f"--train-end={args.train_end} chỉ chừa {train_mask_daily.sum()} phiên, "
            f"quá ít để fit MLE ổn định"
        )
    # Chuẩn hoá biến ngoại sinh CHỈ bằng thống kê của đoạn train (đúng quy ước
    # đã có sẵn của package: standardize_with_train_stats).
    X_scaled, _, _ = standardize_with_train_stats(
        X_raw, train_mask_daily, exclude_cols=("date",)
    )

    valid = (
        daily["log_return"].notna()
        & X_scaled["log_volume_lag1"].notna()
        & X_scaled["log_rv_lag1"].notna()
    )
    dates = daily.loc[valid, "date"].reset_index(drop=True)
    r = torch.tensor(daily.loc[valid, "log_return"].to_numpy())
    X = torch.tensor(X_scaled.loc[valid, ["log_volume_lag1", "log_rv_lag1"]].to_numpy())
    fit_mask = (dates <= train_end).to_numpy()

    print(
        f"Fit MLE trên {fit_mask.sum()} phiên (đến {train_end.date()}); "
        f"bộ lọc tiến sẽ chạy trên toàn bộ {len(r)} phiên "
        f"({dates.iloc[0].date()} -> {dates.iloc[-1].date()})"
    )
    print(
        f"[Lưu ý hiệu năng] n_starts={args.n_starts}, n_adam={args.n_adam}, "
        f"n_lbfgs={args.n_lbfgs} - đây là ngân sách PROTOTYPE (nhẹ hơn cấu hình "
        f"'full' dùng cho bản thảo chính thức), ước vài phút đến ~20 phút trên CPU."
    )

    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
    res = fit_mle(
        r[fit_mask], X[fit_mask], config,
        n_starts=args.n_starts, base_seed=0,
        n_adam=args.n_adam, n_lbfgs=args.n_lbfgs, verbose=True,
    )
    params = label_switch_correction(res.params, res.filter_output["h"])
    print(f"\nloglik={res.loglik:.2f}  AIC={res.aic:.2f}  BIC={res.bic:.2f}")
    d = expected_duration(params.P)
    print(
        f"Thời gian tồn tại kỳ vọng: trạng thái êm={d[0]:.1f} phiên, "
        f"trạng thái động={d[1]:.1f} phiên"
    )
    for run in res.runs:
        print(
            f"  seed={run.seed} loglik={run.loglik:.2f} converged={run.converged} "
            f"grad_norm={run.grad_norm:.4f} near_boundary={run.near_boundary}"
        )
    if any(run.near_boundary for run in res.runs if run.seed == res.runs[0].seed):
        pass  # chẩn đoán near_boundary đã in ở trên, không chặn xuất file

    # Bộ lọc tiến trên TOÀN BỘ mẫu với tham số đã CHỐT trên train (suy luận
    # thuần tuý, không ước lượng lại) - h0 ép bằng phương sai đoạn train để
    # điều kiện khởi tạo không "nhìn" dữ liệu validation/test.
    h0_train = r[fit_mask].var().expand(2).clone()
    with torch.no_grad():
        full_filter = ms_garch_forward_filter(params, r, X, h0=h0_train)

    xi_filt = full_filter["xi_filt"].numpy()  # (T, 2); cột 0 = trạng thái êm
    out = pd.DataFrame({"date": dates, "p_calm_state": xi_filt[:, 0]})
    out.to_csv(args.out, index=False)

    print(f"\nĐã ghi {args.out} ({len(out)} phiên, {time.time() - t_start:.0f}s)")
    print("[!] Tham số MS-GARCH-X ước lượng MỘT LẦN trên đoạn train, KHÔNG walk-forward.")
    print("[!] p_calm_state CHƯA lag ngày trong file này - bên đọc phải tự lag (xem docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
