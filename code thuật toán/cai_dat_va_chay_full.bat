@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

rem --- Tu boc lai trong "cmd /k" NGAY TU DAU: dam bao cua so KHONG THE tu
rem dong lang le du gap loi/crash bat ky o dau trong script (vd. loi cmd.exe
rem noi bo, loi PowerShell, mat ket noi mang giua chung...). Neu double-click
rem lan dau (chua co bien danh dau MSGARCHX_RELAUNCHED), script se tu mo mot
rem cua so cmd MOI voi /k (giu cua so lai sau khi script ket thuc, du thanh
rem cong hay loi) roi thoat cua so goc ngay lap tuc.
if not "%MSGARCHX_RELAUNCHED%"=="1" (
    set "MSGARCHX_RELAUNCHED=1"
    start "GARCH-VN30F1M -- Cai dat va chay FULL" cmd /k "%~f0"
    exit /b
)

if not exist logs mkdir logs
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "TS=%%T"
set "LOGFILE=logs\run_full_%TS%.log"

echo ============================================================
echo  GARCH-VN30F1M -- Cai dat thu vien va chay uoc luong FULL
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
    goto :ketthuc
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
    rem Dau ngoac trong "^(--user^)" PHAI duoc escape bang dau ^: neu de tran
    rem la "(--user)" thi cmd coi dau ) do la dau dong khoi "if errorlevel 1 ("
    rem o tren, khien phan con lai ("...") thanh lenh la va bao
    rem "... was unexpected at this time." roi ABORT ca file .bat.
    echo [Canh bao] Cai dat mac dinh that bai, thu lai voi quyen nguoi dung ^(--user^) ...
    %PYEXE% -m pip install --user -r requirements.txt
    if errorlevel 1 (
        echo [LOI] Cai dat thu vien that bai. Kiem tra ket noi mang, quyen ghi,
        echo       hoac chay thu cmd nay bang tay de xem chi tiet loi:
        echo         %PYEXE% -m pip install -r requirements.txt
        goto :ketthuc
    )
)

echo.
echo Kiem tra file du lieu ohlc_export.csv ...
if not exist "ohlc_export.csv" (
    echo [LOI] Khong tim thay ohlc_export.csv trong thu muc nay: "%cd%"
    echo Hay dat file du lieu VN30F1M vao dung thu muc "code thuat toan" roi chay lai.
    goto :ketthuc
)

echo.
echo [2/2] Bat dau chay uoc luong FULL (MLE, Variational Bayes, Simulation S1-S4).
echo.
echo   THOI GIAN UOC TINH: pipeline chinh tap trung vao MLE va Variational Bayes
echo   (da loai bo MCMC do khong hoi tu), thoi gian chay uoc tinh khoang 3-5 gio.
echo   Ket qua da duoc ghi checkpoint san trong thu muc results\ nen se bo qua
echo   cac buoc da hoan thanh.
echo.
echo   Ket qua duoc ghi checkpoint LIEN TUC vao thu muc results\ -- neu tat may
echo   hoac dong cua so nay giua chung, chay lai file .bat nay se TU DONG bo
echo   qua cac buoc da xong, khong mat tien do.
echo.
echo   LUU Y: trong luc chay, cua so nay co the im lang VAI PHUT giua hai dong
echo   log (dac biet o buoc MLE cua B4/M1) -- do la BINH THUONG, khong phai bi
echo   treo; moi diem khoi tao MLE in tien do dinh ky (moi ~100-200 buoc toi
echo   uu). Neu im lang QUA 30-45 phut khong co dong moi, moi thuc su dang nghi.
echo.
pause

rem Chay THANG ra cua so nay, KHONG qua pipe/PowerShell: giu duoc tien do
rem thoi gian thuc (ke ca thanh tien do tqdm cua MCMC) va tranh loi
rem NativeCommandError cua PowerShell 5.1 khi boc stderr chuong trinh ngoai.
rem "-u" tat buffering de moi dong print hien ra ngay lap tuc.
rem Viec ghi nhat ky do chinh Python dam nhan qua --log-file.
echo Nhat ky day du: %LOGFILE%
echo.
%PYEXE% -u scripts\pipeline\run_full_study.py --preset full --log-file "%LOGFILE%"

:ketthuc
echo.
echo ============================================================
echo  Da dung (hoan tat hoac gap loi). Kiem tra:
echo    - Thu muc results\  (ket qua da checkpoint)
echo    - File nhat ky: %LOGFILE%
echo  Chay lai file nay bat cu luc nao de tiep tuc cac buoc con thieu.
echo.
echo  Cua so nay KHONG tu dong dong (dung cmd /k). Dong bang tay khi da doc
echo  xong, hoac go "exit" roi Enter.
echo ============================================================
