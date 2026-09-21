@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  MS-GARCH-X VN30F1M -- Tao bao cao HTML tu ket qua trong results\
echo ============================================================
echo.

set "PYEXE="
where python >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=python"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYEXE=py"
    )
)

if "%PYEXE%"=="" (
    echo [LOI] Khong tim thay Python trong PATH.
    echo Hay chay file cai_dat_va_chay_full.bat truoc it nhat mot lan.
    pause
    exit /b 1
)

if not exist "results" (
    echo [LOI] Chua co thu muc results\. Hay chay cai_dat_va_chay_full.bat
    rem Dau ngoac PHAI escape bang ^ khi nam trong khoi "if ... ( ... )",
    rem neu khong cmd se coi ) la dau dong khoi va bao loi cu phap roi ABORT.
    echo truoc ^(co the dung khi con dang chay de xem bao cao TAM THOI^).
    pause
    exit /b 1
)

%PYEXE% scripts\tien_ich\build_html_report.py --results-dir results --out report.html
if errorlevel 1 (
    echo [LOI] Tao bao cao that bai. Xem thong bao loi o tren.
    pause
    exit /b 1
)

echo.
echo Da tao report.html. Dang mo bang trinh duyet mac dinh...
start "" "report.html"

echo.
echo (Co the chay lai file nay bat cu luc nao, ke ca khi qua trinh uoc luong
echo  full van dang chay o cua so khac, de xem tien do tam thoi.)
pause
