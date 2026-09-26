@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

rem Chon Python CO torch (co the co nhieu ban Python tren may).
set "PYEXE="
for %%P in (python py) do (
    if not defined PYEXE (
        %%P -c "import torch, pyro" >nul 2>nul && set "PYEXE=%%P"
    )
)
if not defined PYEXE (
    echo [LOI] Khong tim thay Python co torch va pyro. Chay: pip install -r requirements.txt
    pause
    exit /b 1
)

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
