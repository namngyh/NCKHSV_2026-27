@echo off
rem ======================================================================
rem  Duong ong EGARCH-X ky quy dong (bam de cuong 10/2026)
rem  Mo tep nay de chon thao tac. Moi buoc co diem kiem tra: tat may hoac
rem  Ctrl+C bat ky luc nao, chay lai se tiep tuc tu cho dang do.
rem  Thiet ke: THIET_KE_CODE.md. Cau hinh: cau_hinh\de_cuong.toml
rem  Chay song song voi Colab: may nay la "may_manh" (chu tri), Colab chay
rem  notebook colab\chay_tren_colab.ipynb; ket qua dong bo qua GitHub.
rem ======================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PYEXE="
set "KIEM_TRA=import numpy, pandas, scipy, numba, statsmodels, arviz, matplotlib, jax, numpyro, arch, tomllib"

rem Tuy chon: ghi duong dan python.exe vao cau_hinh\python.txt de dung moi truong rieng (vd. conda)
if exist "cau_hinh\python.txt" set /p PYEXE=<"cau_hinh\python.txt"
if defined PYEXE set PYEXE="%PYEXE%"
if not defined PYEXE call :tim_python
if defined PYEXE goto :co_python
echo [INFO] Chua thay Python 3.11 tro len.
where winget >nul 2>nul
if errorlevel 1 (
    echo [LOI] Khong co winget. Hay cai Python 3.11 tu https://www.python.org/downloads/
    echo       ^(tick "Add python.exe to PATH"^) roi mo lai chay.bat.
    pause
    exit /b 1
)
echo [INFO] Dang cai Python 3.11 bang winget ^(can Internet^)...
winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
call :tim_python
if not defined PYEXE (
    echo [INFO] Da cai Python. DONG cua so nay va mo lai chay.bat.
    pause
    exit /b 0
)

:co_python
%PYEXE% -c "%KIEM_TRA%" >nul 2>nul
if not errorlevel 1 goto :menu
echo [INFO] Thieu thu vien. Dang cai dung phien ban trong requirements.txt ...
%PYEXE% -m pip install --upgrade pip
%PYEXE% -m pip install -r requirements.txt
%PYEXE% -c "%KIEM_TRA%" >nul 2>nul
if errorlevel 1 (
    echo [LOI] Cai thu vien chua thanh cong. Xem thong bao o tren.
    pause
    exit /b 1
)

:menu
echo.
echo ==============================================================
echo   EGARCH-X ky quy dong VN30F1M          Python: %PYEXE%
echo ==============================================================
echo   1. Kiem tra moi truong ^(phien ban thu vien, so loi CPU, git^)
echo   2. Chay CHU TRI ^(may manh + Colab, dong bo GitHub^)   ^<- dung khi chay chinh thuc
echo   3. Chay MOT MAY ^(khong Colab, khong dong bo^)
echo   4. Chay LAM DON VI ^(may phu khac, neu co^)
echo   5. Xem tien do ^(lay ket qua moi nhat tu GitHub truoc^)
echo   6. Chay thu NHANH toan bo ^(kiem tra duong ong, khoang 30 phut^)
echo   7. Kiem thu ^(pytest^)
echo   8. Chay mot hoac vai buoc ^(vd: b04 b05^)
echo   9. Xoa va chay lai tu dau mot buoc
echo  10. Sinh lai bao cao va kiem toan ^(b13, b15^)
echo  11. Dong goi nop GVHD ^(b14^)
echo   0. Thoat
echo --------------------------------------------------------------
set "CHON="
set /p CHON="Chon: "
if "%CHON%"=="1" goto :m1
if "%CHON%"=="2" goto :m2
if "%CHON%"=="3" goto :m3
if "%CHON%"=="4" goto :m4
if "%CHON%"=="5" goto :m5
if "%CHON%"=="6" goto :m6
if "%CHON%"=="7" goto :m7
if "%CHON%"=="8" goto :m8
if "%CHON%"=="9" goto :m9
if "%CHON%"=="10" goto :m10
if "%CHON%"=="11" goto :m11
if "%CHON%"=="0" exit /b 0
goto :menu

:m1
%PYEXE% -c "import sys, os, jax, numpyro, numpy, scipy, arch; print('Python', sys.version.split()[0]); print('jax', jax.__version__, 'numpyro', numpyro.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'arch', arch.__version__); print('So loi CPU:', os.cpu_count())"
git --version
git log -1 --format="Commit hien tai: %%h %%s"
goto :xong

:m2
echo [INFO] May nay chay vai tro CHU TRI ^(ten may: may_manh^). Dong cua so nay se dung; mo lai va chon 2 de tiep tuc.
:lap_chu_tri
%PYEXE% quy_trinh\chay.py --buoc tat_ca --vai_tro chu_tri --may may_manh --git
if "%ERRORLEVEL%"=="4" (
    echo [INFO] Ma nguon tren GitHub vua thay doi: khoi dong lai voi ban moi...
    goto :lap_chu_tri
)
call :bao_ma
goto :xong

:m3
%PYEXE% quy_trinh\chay.py --buoc tat_ca
call :bao_ma
goto :xong

:m4
if exist "cau_hinh\ten_may.txt" (set /p TEN_MAY=<"cau_hinh\ten_may.txt") else (
    set "TEN_MAY="
    set /p TEN_MAY="Ten may trong [phan_tan.may] cua cau hinh (vd: may_B): "
    >"cau_hinh\ten_may.txt" echo !TEN_MAY!
)
%PYEXE% quy_trinh\chay.py --buoc tat_ca --vai_tro don_vi --may %TEN_MAY% --git
call :bao_ma
goto :xong

:m5
git pull --rebase --autostash -q
%PYEXE% quy_trinh\chay.py --tien_do
goto :xong

:m6
%PYEXE% quy_trinh\chay.py --buoc tat_ca --ho_so nhanh --chap_nhan_cong
call :bao_ma
goto :xong

:m7
rem Thu muc tam rieng trong repo: tranh loi "Access is denied" o thu muc TEMP\pytest-of-ten_may tren mot so may Windows
if not exist "%~dp0ket_qua_nhanh" mkdir "%~dp0ket_qua_nhanh"
%PYEXE% -m pytest kiem_thu -q -p no:cacheprovider --basetemp="%~dp0ket_qua_nhanh\_pytest_tam"
goto :xong

:m8
set "BUOC="
set /p BUOC="Nhap ma buoc (cach nhau boi dau cach, vd: b04 b05): "
set "CN="
set /p CN="Chu dong chap nhan neu cong khong dat? (y/N): "
set "THEM="
if /i "%CN%"=="y" set "THEM=--chap_nhan_cong"
%PYEXE% quy_trinh\chay.py --buoc %BUOC% %THEM%
call :bao_ma
goto :xong

:m9
set "BUOC="
set /p BUOC="Ma buoc can xoa de chay lai tu dau (vd: b05): "
set "XN="
set /p XN="Xoa ho so, diem kiem tra va dau ra cua %BUOC% (ho so day du)? (y/N): "
if /i not "%XN%"=="y" goto :menu
%PYEXE% quy_trinh\chay.py --xoa %BUOC% --buoc %BUOC%
call :bao_ma
goto :xong

:m10
%PYEXE% quy_trinh\chay.py --buoc b13 b15 --chay_lai
call :bao_ma
goto :xong

:m11
%PYEXE% quy_trinh\chay.py --buoc b14 --chay_lai
call :bao_ma
goto :xong

:bao_ma
if "%ERRORLEVEL%"=="3" echo [CONG] Cong chuyen giai doan KHONG DAT. Xem ket_qua\^<buoc^>\cong.json va trao doi voi GVHD truoc khi chap nhan ^(muc 8^).
if "%ERRORLEVEL%"=="130" echo [DUNG] Da dung. Chay lai se tiep tuc tu diem kiem tra.
if "%ERRORLEVEL%"=="1" echo [LOI] Co loi. Xem thong bao o tren va ket_qua\_nhat_ky\.
exit /b 0

:xong
echo.
pause
goto :menu

:tim_python
for %%P in (python py) do (
    if not defined PYEXE (
        %%P -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>nul && set "PYEXE=%%P"
    )
)
exit /b 0
