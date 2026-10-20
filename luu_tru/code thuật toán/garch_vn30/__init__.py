"""garch_vn30: uoc luong va danh gia ho mo hinh GARCH/EGARCH cho VN30F1M.

Ba phuong phap suy dien (MLE, Variational Bayes, MCMC) dung CUNG MOT ham
log-hop-ly cho moi mo hinh. Cau truc thu vien:

    du_lieu/     doc du lieu phut, gop ngay, loi suat, RV, bien ngoai sinh
    kiem_dinh/   kiem dinh thong ke ban dau (ADF, PP, KPSS, Ljung-Box, ARCH-LM, JB)
    mo_hinh/     phan phoi sai so, GARCH(-X), EGARCH(-X), danh muc 6 mo hinh
    suy_dien/    MLE, mo hinh Bayes (Pyro), VB mean-field/full-rank, MCMC (NUTS)
    danh_gia/    danh gia ngoai mau: QLIKE, NLS, VaR (Kupiec, Christoffersen), DM
    mo_phong/    nghien cuu khoi phuc tham so cho EGARCH-X Student-t
"""
import sys as _sys

import torch as _torch

# Moi phep tinh dung float64: vong lap phuong sai chay qua ~1.700 quan sat va
# Hessian can do chinh xac cao; float32 lam sai lech log-hop-ly o chu so thu 4.
_torch.set_default_dtype(_torch.float64)

# Console Windows mac dinh dung cp1252: ep UTF-8 de in duoc chu tieng Viet.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

__version__ = "1.0.0"
