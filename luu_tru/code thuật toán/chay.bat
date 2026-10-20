@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYEXE="
set "KIEM_TRA_THU_VIEN=import torch, pyro, numpy, pandas, scipy, statsmodels, arch, arviz, matplotlib, pytest"

rem ---- Buoc 1: tim Python 3.10 tro len; neu chua co thi cai bang winget ----
call :tim_python
if defined PYEXE goto :co_python

echo [INFO] Chua thay Python 3.10+ tren may.
where winget >nul 2>nul
if errorlevel 1 (
    echo [LOI] Khong co winget. Hay cai Python 3.11 tu https://www.python.org/downloads/
    echo       ^(nho tick "Add python.exe to PATH"^) roi mo lai chay.bat.
    pause
    exit /b 1
)
echo [INFO] Dang cai Python 3.11 bang winget ^(can Internet, vai phut^)...
winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
call :tim_python
if not defined PYEXE (
    echo [INFO] Da cai xong Python. Hay DONG cua so nay va mo lai chay.bat de nhan Python moi.
    pause
    exit /b 0
)

:co_python
rem ---- Buoc 2: kiem tra thu vien; chi cai neu con thieu ----
%PYEXE% -c "%KIEM_TRA_THU_VIEN%" >nul 2>nul
if not errorlevel 1 goto :menu
echo [INFO] Thieu thu vien. Dang cai tu requirements.txt ^(torch kha nang, co the lau^)...
%PYEXE% -m pip install --upgrade pip
%PYEXE% -m pip install -r requirements.txt
%PYEXE% -c "%KIEM_TRA_THU_VIEN%" >nul 2>nul
if errorlevel 1 (
    echo [LOI] Cai thu vien chua thanh cong. Xem thong bao loi o tren.
    pause
    exit /b 1
)
echo [OK] Da cai du thu vien.
goto :menu

:tim_python
for %%P in (python py) do (
    if not defined PYEXE (
        %%P -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>nul && set "PYEXE=%%P"
    )
)
exit /b 0

:menu
echo.
echo ==============================================================
echo   GARCH-VN30F1M-MLE-VB-MCMC        (Python: %PYEXE%)
echo ==============================================================
echo   1. Kiem tra: chay test + kiem toan ket qua (vai phut)
echo   2. Buoc 1: so sanh 6 mo hinh x 3 phuong phap (nhieu gio)
echo   3. Buoc 2: mo phong khoi phuc tham so EGARCH-X (khoang 30 gio, 8 tien trinh)
echo   4. Buoc 3: MCMC cho mo hinh thang cuoc (khoang 8 gio)
echo   5. Sinh lai hinh va file tong_hop_ket_qua.tex
echo   6. Chay thu nhanh ca ba buoc (--nhanh, vai phut)
echo   0. Thoat
echo.
set /p CHON="Chon: "
if "%CHON%"=="1" (
    %PYEXE% -m pytest tests -q
    %PYEXE% scripts\kiem_toan_ket_qua.py
)
if "%CHON%"=="2" %PYEXE% scripts\chay_buoc1_so_sanh_mo_hinh.py
if "%CHON%"=="3" %PYEXE% scripts\chay_buoc2_mo_phong.py --jobs 8
if "%CHON%"=="4" %PYEXE% scripts\chay_buoc3_mcmc.py
if "%CHON%"=="5" (
    %PYEXE% scripts\bao_cao\ve_vector_tham_so.py
    %PYEXE% scripts\bao_cao\sinh_hinh_tong_hop.py
    %PYEXE% scripts\bao_cao\sinh_tong_hop_ket_qua.py
)
if "%CHON%"=="6" (
    %PYEXE% scripts\chay_buoc1_so_sanh_mo_hinh.py --nhanh
    %PYEXE% scripts\chay_buoc2_mo_phong.py --nhanh --jobs 4
    %PYEXE% scripts\chay_buoc3_mcmc.py --mo-hinh egarch_x_t --nhanh
)
if "%CHON%"=="0" goto :het
goto :menu

:het
endlocal
