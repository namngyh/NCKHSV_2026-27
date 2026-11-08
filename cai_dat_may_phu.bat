@echo off
rem ======================================================================
rem  CAI DAT VA CHAY MAY PHU (mot lan nhap dup la xong)
rem  Chep RIENG tep nay sang may moi (USB, Zalo, Drive...), dat o Desktop
rem  roi nhap dup. Tep se tu:
rem    1. cai Git va Python 3.11 neu chua co (winget, can Internet)
rem    2. tai repo ve thu muc NCKHSV_2026-27 canh tep nay (khoang vai GB)
rem    3. tao moi truong .venv va cai dung phien ban trong requirements.txt
rem    4. ghi ten may vao cau_hinh\ten_may.txt (mac dinh may_phu_2)
rem    5. kiem tra ma bam b03a/b03b khop may chu tri (lech thi DUNG)
rem    6. chay vai tro "lam don vi" va dong bo ket qua qua GitHub
rem  Viec thu cong duy nhat: dang nhap GitHub khi cua so dang nhap hien ra
rem  (tai khoan co quyen ghi repo namngyh/NCKHSV_2026-27).
rem  Chay lai tep nay bat ky luc nao: buoc da xong se duoc bo qua, phan tinh
rem  toan tiep tuc tu diem kiem tra. Ten may khac: cai_dat_may_phu.bat may_phu_3
rem ======================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Cai dat may phu NCKHSV
set "REPO_URL=https://github.com/namngyh/NCKHSV_2026-27.git"
set "TEN_MAY=%~1"
if not defined TEN_MAY set "TEN_MAY=may_phu_2"

rem Tep dat ngay trong repo thi dung repo do; neu khong thi tai repo ve canh tep nay
set "GOC=%~dp0"
set "GOC=%GOC:~0,-1%"
if not exist "%GOC%\quy_trinh\chay.py" set "GOC=%GOC%\NCKHSV_2026-27"
echo [INFO] Ten may: %TEN_MAY%
echo [INFO] Thu muc repo: %GOC%
echo.

rem ---------------------------------------------------------------- 1. Git
echo [1/6] Kiem tra Git...
where git >nul 2>nul
if not errorlevel 1 goto :co_git
call :can_winget || goto :that_bai
echo [INFO] Dang cai Git ^(co the hien hop thoai cap quyen, chon Yes^)...
winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
where git >nul 2>nul
if errorlevel 1 (
    echo [INFO] Da cai Git nhung cua so nay chua nhan. DONG cua so va nhap dup lai tep nay.
    goto :that_bai
)
:co_git
rem Cung kieu xuong dong voi may chu tri (CRLF) - neu khac, ma bam se lech
git config --global core.autocrlf true
git config --global core.longpaths true

rem ---------------------------------------------------------------- 2. Repo
echo [2/6] Tai repo...
if exist "%GOC%\quy_trinh\chay.py" goto :co_repo
if exist "%GOC%\.git" (
    echo [INFO] Thay ban tai do dang ^(lan truoc bi ngat^). Xoa va tai lai...
    rd /s /q "%GOC%"
)
echo [INFO] Dang tai repo, co the mat 10-60 phut tuy mang. Neu hien cua so dang nhap GitHub, hay dang nhap.
git -c core.autocrlf=true clone --depth 1 --single-branch --branch main "%REPO_URL%" "%GOC%"
if not exist "%GOC%\quy_trinh\chay.py" (
    echo [LOI] Tai repo khong thanh cong. Kiem tra mang va quyen truy cap repo roi nhap dup lai tep nay.
    goto :that_bai
)
goto :cau_hinh_git
:co_repo
git -C "%GOC%" pull --rebase --autostash -q
:cau_hinh_git
git -C "%GOC%" config core.autocrlf true
rem Danh tinh commit rieng cua may nay (khong dung email ca nhan)
git -C "%GOC%" config user.name "%TEN_MAY%"
git -C "%GOC%" config user.email "%TEN_MAY%@may-phu.local"
echo [INFO] Kiem tra quyen ghi GitHub ^(dang nhap neu duoc hoi^)...
git -C "%GOC%" push --dry-run -q >nul
if errorlevel 1 (
    echo [LOI] Chua co quyen ghi len GitHub. Dang nhap bang tai khoan co quyen ghi repo roi nhap dup lai tep nay.
    goto :that_bai
)

rem ---------------------------------------------------------------- 3. Python
echo [3/6] Kiem tra Python 3.11...
set "PYEXE=%GOC%\.venv\Scripts\python.exe"
if exist "%PYEXE%" goto :co_venv
call :tim_py311
if defined PY311 goto :tao_venv
call :can_winget || goto :that_bai
echo [INFO] Dang cai Python 3.11...
winget install --id Python.Python.3.11 -e --silent --scope user --accept-package-agreements --accept-source-agreements
call :tim_py311
if not defined PY311 (
    echo [INFO] Da cai Python nhung chua tim thay. DONG cua so va nhap dup lai tep nay.
    goto :that_bai
)
:tao_venv
echo [INFO] Tao moi truong rieng .venv tu %PY311%
"%PY311%" -m venv "%GOC%\.venv"
:co_venv
set "KIEM_TRA=import numpy, pandas, scipy, numba, statsmodels, arviz, matplotlib, jax, numpyro, arch, tomllib"
"%PYEXE%" -c "%KIEM_TRA%" >nul 2>nul
if not errorlevel 1 goto :du_thu_vien
echo [INFO] Dang cai thu vien dung phien ban trong requirements.txt ^(vai phut^)...
"%PYEXE%" -m pip install --upgrade pip -q
"%PYEXE%" -m pip install -r "%GOC%\requirements.txt"
"%PYEXE%" -c "%KIEM_TRA%" >nul 2>nul
if errorlevel 1 (
    echo [LOI] Cai thu vien chua thanh cong. Xem thong bao o tren.
    goto :that_bai
)
:du_thu_vien
>"%GOC%\cau_hinh\python.txt" echo %PYEXE%

rem ---------------------------------------------------------------- 4. Ten may
echo [4/6] Ghi ten may...
findstr /b /c:"%TEN_MAY% =" "%GOC%\cau_hinh\de_cuong.toml" >nul
if errorlevel 1 (
    echo [LOI] Cau hinh [phan_tan.may] chua co may "%TEN_MAY%". Nho Claude/nguoi chu tri them vao roi chay lai.
    goto :that_bai
)
>"%GOC%\cau_hinh\ten_may.txt" echo %TEN_MAY%

rem ---------------------------------------------------------------- 5. Ma bam
echo [5/6] Kiem tra ma bam ^(phai KHOP ca hai dong^)...
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%PYEXE%" -X utf8 "%GOC%\quy_trinh\kiem_ma_bam.py" %TEN_MAY%
if errorlevel 1 (
    echo [LOI] Ma bam LECH. KHONG chay de tranh tinh sai thu muc. Chup man hinh nay gui Claude.
    goto :that_bai
)

rem ---------------------------------------------------------------- 6. Chay
echo [6/6] Bat dau chay. Co the thu nho cua so; DUNG tat may hay de may ngu.
echo       Dung: Ctrl+C. Chay tiep: nhap dup lai tep nay.
rem Khong cho may ngu khi cam sac (dat lai: Settings ^> System ^> Power)
powercfg /change standby-timeout-ac 0 >nul 2>nul
cd /d "%GOC%"
:lap
"%PYEXE%" quy_trinh\chay.py --buoc tat_ca --vai_tro don_vi --may %TEN_MAY% --git
set "MA=%ERRORLEVEL%"
if "%MA%"=="4" (
    echo [INFO] Ma nguon tren GitHub vua thay doi: khoi dong lai voi ban moi...
    goto :lap
)
if "%MA%"=="130" echo [DUNG] Da dung. Nhap dup lai tep nay de chay tiep tu diem kiem tra.
if "%MA%"=="1" echo [LOI] Co loi. Xem thong bao o tren va ket_qua\_nhat_ky\.
if "%MA%"=="0" echo [XONG] Phan viec cua may nay da xong.
pause
exit /b 0

:that_bai
echo.
pause
exit /b 1

:can_winget
where winget >nul 2>nul
if not errorlevel 1 exit /b 0
echo [LOI] May chua co winget. Cai "App Installer" tu Microsoft Store roi nhap dup lai tep nay.
exit /b 1

:tim_py311
set "PY311="
for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PY311=%%P"
if defined PY311 exit /b 0
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%ProgramFiles%\Python311\python.exe") do (
    if not defined PY311 if exist %%P set "PY311=%%~P"
)
exit /b 0
