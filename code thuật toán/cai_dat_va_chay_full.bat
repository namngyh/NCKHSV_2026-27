@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  MS-GARCH-X VN30F1M -- Cai dat thu vien va chay uoc luong FULL
echo ============================================================
echo.

rem --- Tim trinh thong dich Python (uu tien "python", du phong "py") ---
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
    echo Hay cai Python 3.10+ tu https://www.python.org/downloads/ va nho tich
    echo chon "Add python.exe to PATH" luc cai dat, roi chay lai file nay.
    pause
    exit /b 1
)

echo Dung trinh thong dich: %PYEXE%
%PYEXE% --version
echo.

echo [1/2] Cai dat thu vien tu requirements.txt (co the mat vai phut, tai ve
echo       PyTorch/Pyro kha nang).
echo.
%PYEXE% -m pip install --upgrade pip
%PYEXE% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [Canh bao] Cai dat mac dinh that bai, thu lai voi quyen nguoi dung (--user) ...
    %PYEXE% -m pip install --user -r requirements.txt
    if errorlevel 1 (
        echo [LOI] Cai dat thu vien that bai. Kiem tra ket noi mang, quyen ghi,
        echo       hoac chay thu cmd nay bang tay de xem chi tiet loi:
        echo         %PYEXE% -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo Kiem tra file du lieu ohlc_export.csv ...
if not exist "ohlc_export.csv" (
    echo [LOI] Khong tim thay ohlc_export.csv trong thu muc nay: "%cd%"
    echo Hay dat file du lieu VN30F1M vao dung thu muc "code thuat toan" roi chay lai.
    pause
    exit /b 1
)

echo.
echo [2/2] Bat dau chay uoc luong FULL.
echo.
echo   CANH BAO THOI GIAN: cau hinh "full" co the mat 1.5-2+ NGAY (dac biet la
echo   MCMC tren du lieu that, xem "Luu y hieu nang" trong README.md). May tinh
echo   can duoc bat va KHONG ngu (sleep) lien tuc trong qua trinh nay.
echo.
echo   Ket qua duoc ghi checkpoint LIEN TUC vao thu muc results\ -- neu tat may
echo   hoac dong cua so nay giua chung, chay lai file .bat nay se TU DONG bo
echo   qua cac buoc da xong, khong mat tien do.
echo.
echo   Nhat ky day du duoc luu vao thu muc logs\ (va van hien tren man hinh nay).
echo.
pause

if not exist logs mkdir logs

rem Lay timestamp qua PowerShell (khong dung %date%/%time% vi dinh dang phu
rem thuoc locale Windows, co the sai voi Windows tieng Viet).
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "TS=%%T"
set "LOGFILE=logs\run_full_%TS%.log"

echo Nhat ky: %LOGFILE%
echo.

rem Chi dung dau nhay don ben trong chuoi -Command de tranh loi escape dau
rem ngoac kep long nhau cua cmd.exe (cmd KHONG ho tro \" nhu POSIX shell).
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PYEXE%' scripts\run_full_study.py --preset full 2>&1 | Tee-Object -FilePath '%LOGFILE%'"

echo.
echo ============================================================
echo  Da dung (hoan tat hoac gap loi). Kiem tra:
echo    - Thu muc results\  (ket qua da checkpoint)
echo    - Thu muc logs\     (nhat ky day du)
echo  Chay lai file nay bat cu luc nao de tiep tuc cac buoc con thieu.
echo ============================================================
pause
