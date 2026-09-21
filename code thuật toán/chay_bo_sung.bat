@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "JOBS=12"
title Bo sung coverage cho MLE va Brier dung cho PR-A

rem Chon interpreter CO TORCH, khong phai cai co ten "python": PATH may nay
rem tro `python` toi Python 3.12 khong co torch (cai cho du an khac), con
rem ban co torch la 3.13 qua `py`. Chon theo ten se chon nham.
set "PYEXE="
py -c "import torch" >nul 2>nul
if not errorlevel 1 set "PYEXE=py"
if not defined PYEXE python -c "import torch" >nul 2>nul
if not defined PYEXE if not errorlevel 1 set "PYEXE=python"
if not defined PYEXE goto :khong_co_python

cls
echo ================================================================
echo  BO SUNG HAI DAI LUONG CON THIEU -- KHONG uoc luong lai
echo ================================================================
echo.
echo  1. COVERAGE CHO MLE (PR-A va PR-B)
echo     Bang so sanh dang thien vi VB o cot coverage chi vi MLE chua
echo     duoc tinh. Nay dung sai so chuan tu Hessian tai nghiem DA LUU,
echo     khoang Wald theta +/- 1,96 SE. Hessian co the khong xac dinh
echo     duong o nghiem sat bien -- khi do ghi "hong" va DEM, khong coi
echo     la 0.
echo.
echo  2. BRIER DUNG CHO PR-A
echo     Brier cu tinh TRUOC khi sap nhan trang thai nen sai (VB-FR
echo     0,63-0,68, te hon doan ngau nhien 0,50). Nay sinh lai du lieu
echo     cung seed, sap nhan tham so da luu, chay lai BO LOC (khong toi
echo     uu hoa), cham so voi trang thai that.
echo.
echo  Khoi luong: 900 (PR-A) + 300 (PR-B) = 1200 tac vu, %JOBS% tien trinh.
echo  Do duoc tren may nay: MLE PR-A ~39 giay (co Hessian 15 chieu),
echo  VB PR-A ~1 giay (chi bo loc), MLE PR-B ~12 giay. Uoc tinh ~25-30 phut.
echo.
echo  Checkpoint tung tac vu trong results\parts\bosung\ -- Ctrl+C roi
echo  chay lai se tiep tuc. Ket qua ghep vao 07_simulation_S*.pt va
echo  mo_phong_egarch\E*.pt; ban goc duoc sao luu truoc khi ghi de.
echo.
pause

echo.
%PYEXE% scripts\pipeline\bo_sung_coverage_brier.py --jobs %JOBS%

echo.
echo ================================================================
echo  KET THUC luc %TIME%
echo  Xem ket qua: %PYEXE% scripts\tien_ich\tom_tat_mo_phong.py
echo  Sinh lai bang cho bai: %PYEXE% scripts\bao_cao\sinh_ket_qua_moi.py
echo ================================================================
pause
goto :het

:khong_co_python
echo [LOI] Khong tim thay Python nao co torch.
pause

:het
endlocal
