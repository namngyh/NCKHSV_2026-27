@echo off
rem ======================================================================
rem  MAY PHU: cai dat va chay mo phong b03b chi bang mot lan nhap dup.
rem  - Lan dau: tu cai Git, Python 3.11 va thu vien neu thieu; neu tep nay
rem    duoc tai rieng (khong nam trong repo) thi tu tai repo ve
rem    C:\NCKHSV_2026-27; hoi so may phu mot lan; kiem tra ma bam; dang
rem    nhap GitHub (cua so dang nhap hien ra mot lan).
rem  - Sau do chay lien tuc: tu khoi dong lai khi ma nguon tren GitHub doi,
rem    tu thu lai khi mat mang hoac loi git. Dung: dong cua so.
rem    Mo lai tep nay de chay tiep tu diem kiem tra, khong mat ket qua.
rem  Tep nay tu chep ra thu muc TEMP roi chay ban sao, de lan git pull cap
rem  nhat chinh tep nay khong lam hong cua so dang chay.
rem ======================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
set "REPO_URL=https://github.com/namngyh/NCKHSV_2026-27.git"
set "THU_MUC_MAC_DINH=C:\NCKHSV_2026-27"
if /i "%~1"=="--ban-sao" goto :ban_sao

rem ---------------------------------------------------------------- 1. Git
where git >nul 2>nul
if not errorlevel 1 goto :co_git
echo [1/6] Chua co Git. Dang cai bang winget...
winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
set "PATH=%PATH%;C:\Program Files\Git\cmd"
where git >nul 2>nul
if not errorlevel 1 goto :co_git
echo [LOI] Chua cai duoc Git. Cai tu https://git-scm.com/download/win roi mo lai tep nay.
pause
exit /b 1
:co_git

rem ---------------------------------------------------------------- 2. Repo
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
if exist "%REPO%\quy_trinh\chay.py" goto :co_repo
set "REPO=%THU_MUC_MAC_DINH%"
if exist "%REPO%\quy_trinh\chay.py" goto :co_repo
echo [2/6] Dang tai repo ve %REPO% - lan dau mat khoang 10-30 phut tuy mang...
git clone -c core.longpaths=true -c core.autocrlf=true --depth 1 "%REPO_URL%" "%REPO%"
if errorlevel 1 (
    echo [LOI] Tai repo that bai. Kiem tra mang va quyen truy cap GitHub roi mo lai tep nay.
    pause
    exit /b 1
)
:co_repo
set "NGUON=%REPO%\may_phu.bat"
if not exist "%NGUON%" set "NGUON=%~f0"
set "BAN_SAO=%TEMP%\may_phu_dang_chay_%RANDOM%.bat"
copy /y "%NGUON%" "%BAN_SAO%" >nul
call "%BAN_SAO%" --ban-sao "%REPO%"
del "%BAN_SAO%" >nul 2>nul
exit /b

rem ======================================================================
:ban_sao
cd /d "%~2"
title May phu - NCKHSV b03b
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

rem ---------------------------------------------------------------- 3. Cau hinh git, cap nhat ma moi
git config core.longpaths true
git config core.autocrlf true
git config user.name >nul 2>nul || git config user.name "%COMPUTERNAME%"
git config user.email >nul 2>nul || git config user.email "%COMPUTERNAME%@may-phu.local"
git rebase --abort >nul 2>nul
git merge --abort >nul 2>nul
echo [3/6] Lay ma va ket qua moi nhat tu GitHub...
git pull --rebase --autostash -q
if errorlevel 1 (
    git rebase --abort >nul 2>nul
    git pull --no-rebase --no-edit --autostash -X theirs -q
)
rem May khong duoc ngu khi dang chay (ngu la dung tinh toan)
powercfg /change standby-timeout-ac 0 >nul 2>nul
powercfg /change hibernate-timeout-ac 0 >nul 2>nul

rem ---------------------------------------------------------------- 4. Python va thu vien
set "PYEXE="
set "KIEM_TRA=import numpy, pandas, scipy, numba, statsmodels, arviz, matplotlib, jax, numpyro, arch, tomllib"
if exist "cau_hinh\python.txt" set /p PYEXE=<"cau_hinh\python.txt"
if defined PYEXE set PYEXE="%PYEXE%"
if not defined PYEXE call :tim_python
if defined PYEXE goto :co_python
echo [4/6] Chua co Python 3.11. Dang cai bang winget...
winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PYEXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYEXE call :tim_python
if defined PYEXE goto :co_python
echo [LOI] Chua tim thay Python sau khi cai. Dong cua so nay va mo lai tep may_phu.bat.
pause
exit /b 1
:co_python
%PYEXE% -c "%KIEM_TRA%" >nul 2>nul
if not errorlevel 1 goto :du_thu_vien
echo [4/6] Dang cai thu vien theo requirements.txt - lan dau mat vai phut...
%PYEXE% -m pip install --upgrade pip -q
%PYEXE% -m pip install -r requirements.txt
%PYEXE% -c "%KIEM_TRA%" >nul 2>nul
if not errorlevel 1 goto :du_thu_vien
echo [LOI] Cai thu vien chua thanh cong. Chup man hinh thong bao o tren gui nhom.
pause
exit /b 1
:du_thu_vien

rem ---------------------------------------------------------------- 5. Ten may va kiem tra
set "DA_SUA_XUONG_DONG="
:hoi_ten
set "TEN_MAY="
if exist "cau_hinh\ten_may.txt" set /p TEN_MAY=<"cau_hinh\ten_may.txt"
if defined TEN_MAY goto :kiem_tra
echo.
echo   Day la MAY PHU SO MAY? Nhom truong da phan cong: 1, 2 hoac 3.
set "SO="
set /p SO="  Nhap so roi Enter: "
set "TEN_MAY=may_phu_%SO%"
:kiem_tra
echo [5/6] Kiem tra ten may !TEN_MAY! va ma bam...
%PYEXE% quy_trinh\kiem_tra_may_phu.py !TEN_MAY!
set "MA=%ERRORLEVEL%"
if "%MA%"=="0" goto :ten_dat
if exist "cau_hinh\ten_may.txt" del "cau_hinh\ten_may.txt"
if "%MA%"=="2" goto :hoi_ten
if "%MA%"=="3" if not defined DA_SUA_XUONG_DONG goto :sua_xuong_dong
echo [LOI] Kiem tra khong dat ^(ma %MA%^). Chup man hinh gui nhom.
pause
exit /b 1
:sua_xuong_dong
echo [5/6] Ma bam lech do xuong dong. Dang lay lai ma nguon dang CRLF nhu cac may khac...
set "DA_SUA_XUONG_DONG=1"
git rm -r --cached -q -- ma_nguon quy_trinh du_lieu cau_hinh/de_cuong.toml cau_hinh/the_che.toml
git checkout HEAD -- ma_nguon quy_trinh du_lieu cau_hinh/de_cuong.toml cau_hinh/the_che.toml
goto :kiem_tra
:ten_dat
>"cau_hinh\ten_may.txt" echo !TEN_MAY!

rem ---------------------------------------------------------------- 6. Quyen ghi GitHub
echo [6/6] Kiem tra quyen ghi GitHub - neu hien cua so dang nhap, hay dang nhap tai khoan da duoc moi vao repo...
git push --dry-run -q origin HEAD:main
if not errorlevel 1 goto :chay_dau
echo [LOI] Tai khoan GitHub tren may nay chua co quyen ghi vao repo.
echo       Nho chu repo moi tai khoan cua ban ^(Settings - Collaborators^), chap nhan loi moi trong email,
echo       roi mo lai tep nay.
pause
exit /b 1

rem ---------------------------------------------------------------- Chay lien tuc
:chay_dau
echo.
echo ==============================================================
echo   MAY !TEN_MAY! BAT DAU CHAY. De nguyen cua so nay, KHONG tat may.
echo   Ket qua tu dong day len GitHub moi 30 phut.
echo ==============================================================
:chay
%PYEXE% quy_trinh\chay.py --buoc tat_ca --vai_tro don_vi --may !TEN_MAY! --git
set "MA=%ERRORLEVEL%"
if "%MA%"=="130" exit /b 0
if "%MA%"=="4" goto :khoi_dong_lai
if "%MA%"=="0" goto :tam_het_viec
echo [CANH BAO] Tien trinh dung voi ma %MA%. Tu chay lai sau 2 phut ^(ket qua da xong van duoc giu^)...
timeout /t 120 /nobreak >nul
git rebase --abort >nul 2>nul
git merge --abort >nul 2>nul
goto :chay
:khoi_dong_lai
echo [INFO] Ma nguon tren GitHub vua doi: khoi dong lai ngay voi ban moi...
goto :chay
:tam_het_viec
echo [INFO] Tam het viec cua may nay. Cho 10 phut roi kiem tra lai - dong cua so khi nhom bao b03b da xong.
timeout /t 600 /nobreak >nul
goto :chay

:tim_python
for %%P in (python py) do (
    if not defined PYEXE (
        %%P -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>nul && set "PYEXE=%%P"
    )
)
exit /b 0
