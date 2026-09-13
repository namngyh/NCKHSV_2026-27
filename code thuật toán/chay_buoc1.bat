@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
title Buoc 1 -- so sanh toan bo danh muc mo hinh

set "PYEXE="
where python >nul 2>nul
if not errorlevel 1 set "PYEXE=python"
if not defined PYEXE where py >nul 2>nul
if not defined PYEXE if not errorlevel 1 set "PYEXE=py"
if not defined PYEXE goto :khong_co_python

echo ================================================================
echo  BUOC 1 -- 8 mo hinh x 3 phuong phap tren du lieu VN30F1M that
echo ================================================================
echo.
echo  Luoi mo hinh:
echo                 chuan          Student-t      Student-t + X
echo     GARCH       garch_n  (4)   garch_t  (5)   garch_x_t  (7)
echo     EGARCH      egarch_n (5)   egarch_t (6)   egarch_x_t (8)
echo   cong hai mo hinh chuyen trang thai: ms_garch_n (10), ms_garch_x_t (15)
echo.
echo  Chay TUAN TU mot luong: 17 muc con thieu, uoc tinh khoang 8 gio.
echo  (hai muc VB cho MS-GARCH chuan chiem ~2,6 gio trong so do)
echo  Tien do hien ra ngay tren cua so nay.
echo.
echo  Moi cap (mo hinh, phuong phap) co checkpoint rieng trong
echo  results\pipeline\ -- Ctrl+C dung giua chung roi chay lai se tu bo
echo  qua phan da xong, khong mat gi ngoai muc dang do.
echo.
echo  Dung truoc Buoc 3 (MCMC). Muon chay ca MCMC thi bo --bo-qua-mcmc
echo  trong file .bat nay -- nhung xem ghi chu trong CLAUDE.md truoc.
echo.
pause

echo.
%PYEXE% scripts\pipeline_all_models.py --bo-qua-mcmc

echo.
echo ================================================================
echo  KET THUC luc %TIME%
echo  Bang so sanh o tren; checkpoint trong results\pipeline\
echo  Xem lai bang bat cu luc nao (khong tinh toan lai):
echo      %PYEXE% scripts\pipeline_all_models.py --chi-bang
echo ================================================================
pause
goto :het

:khong_co_python
echo [LOI] Khong tim thay Python trong PATH.
pause

:het
endlocal
