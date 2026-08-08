"""MS-GARCH-X Student-t: uoc luong va du bao bien dong VN30F1M.

Package nay hien thuc hoa khung mo hinh va suy dien mo ta trong bai bao
"Uoc luong va du bao bien dong theo trang thai thi truong bang mo hinh
MS-GARCH-X voi sai so Student-t". Cau truc thu muc:

- ``distributions``: mat do Student-t chuan hoa.
- ``transforms``: bien doi tham so khong rang buoc <-> mien hop le.
- ``models``: GARCH-X mot trang thai va MS-GARCH-X (bo loc tien, log-likelihood).
- ``inference``: MLE, Variational Bayes, MCMC tren cung mot ham hop ly.
- ``evaluation``: ham mat mat bien dong, diem so mat do, kiem dinh VaR.
- ``stats_tests``: kiem dinh thong ke ban dau (ADF, KPSS, Ljung-Box, ARCH-LM, JB).
- ``simulation``: co che sinh du lieu va nghien cuu mo phong S1-S8.
- ``data``: xay dung chuoi hop dong lien tuc, lam sach, bien ngoai sinh.
"""

import sys as _sys

import torch as _torch

# Log-likelihood MS-GARCH nhay voi sai so lam tron (logsumexp, ty le phuong
# sai giua hai trang thai); dung float64 mac dinh cho toan bo package.
_torch.set_default_dtype(_torch.float64)

# Console Windows mac dinh dung cp1252, gay UnicodeEncodeError khi in duong
# dan/chu co dau (vd. thu muc "Bài báo", "code thuật toán"). Ep stdout/stderr
# sang UTF-8 ngay khi import package, khong phu thuoc bien moi truong
# PYTHONIOENCODING cua nguoi goi.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

__version__ = "0.1.0"
