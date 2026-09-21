@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

goto :menu

:tim_python
set "PYEXE="
where python >nul 2>nul
if errorlevel 1 goto :tim_python_thu_py
set "PYEXE=python"
goto :eof

:tim_python_thu_py
where py >nul 2>nul
if errorlevel 1 goto :tim_python_khong_thay
set "PYEXE=py"
goto :eof

:tim_python_khong_thay
echo [LOI] Khong tim thay Python trong PATH.
echo Hay chay cai_dat_va_chay_full.bat truoc it nhat mot lan, hoac cai Python
echo va nho tich "Add python.exe to PATH".
goto :eof

:menu
cls
echo ============================================================
echo  MS-GARCH-X VN30F1M -- Chon cach chay
echo ============================================================
echo.
echo   1. Chay TOAN BO pipeline chinh (data/prelim/MLE/VB/R1/OOS/
echo      Simulation S1-S4) mot luot, trong CUA SO RIENG.
echo      -- Uoc tinh 3-5 gio (tap trung vao MLE va Variational Bayes).
echo.
echo   2. Chay MOT BUOC cua pipeline chinh (vd: chi "mle", "vb", "simulation").
echo      Cac buoc hop le: data prelim mle vb r1 oos simulation
echo.
echo   3. Chay MOT MO HINH rieng le trong ke hoach so sanh 20 mo hinh
echo      (01-07, 12-14 dang san sang -- xem --list de biet chi tiet).
echo.
echo   4. Tao/cap nhat bao cao HTML tong hop tu ket qua da co trong results\
echo      (tu dong ghi "chua co du lieu" cho buoc nao chua chay xong).
echo.
echo   0. Thoat
echo.
set /p "CHOICE=Nhap so lua chon: "

if "%CHOICE%"=="1" goto :chay_full
if "%CHOICE%"=="2" goto :chay_mot_buoc
if "%CHOICE%"=="3" goto :chay_mot_mo_hinh
if "%CHOICE%"=="4" goto :tao_bao_cao
if "%CHOICE%"=="0" goto :ketthuc
echo.
echo Lua chon khong hop le, thu lai.
pause >nul
goto :menu

:chay_full
echo.
echo Dang mo cua so rieng de chay TOAN BO pipeline (cai_dat_va_chay_full.bat) ...
echo Cua so do se tu kiem tra/cai thu vien va chay run_full_study.py --preset full.
start "" "%~dp0cai_dat_va_chay_full.bat"
echo Da mo cua so moi. Quay lai menu ...
pause >nul
goto :menu

:chay_mot_buoc
call :tim_python
if "%PYEXE%"=="" pause
if "%PYEXE%"=="" goto :menu
echo.
echo Cac buoc hop le: data prelim mle vb r1 oos simulation
echo (co the go nhieu buoc cach nhau boi dau cach, vd: mle vb r1)
set /p "STAGE=Nhap TEN BUOC muon chay: "
if "%STAGE%"=="" goto :menu
echo.
echo Dang chay: %PYEXE% -u scripts\pipeline\run_full_study.py --preset full --only %STAGE%
echo (Ket qua checkpoint vao results\, co the dong cua so va chay tiep sau.)
echo.
%PYEXE% -u scripts\pipeline\run_full_study.py --preset full --only %STAGE%
echo.
echo Da xong (hoac gap loi, xem thong bao o tren).
pause >nul
goto :menu

:chay_mot_mo_hinh
call :tim_python
if "%PYEXE%"=="" pause
if "%PYEXE%"=="" goto :menu
echo.
%PYEXE% scripts\pipeline\run_one_model.py --list
echo.
set /p "MID=Nhap SO MO HINH muon chay (vd: 5), de trong de huy: "
if "%MID%"=="" goto :menu
echo.
echo Dang chay: %PYEXE% -u scripts\pipeline\run_one_model.py %MID%
echo (Tien do in truc tiep o duoi; ket qua tu dong ghi them vao
echo  results\screening_log.txt, khong xoa noi dung cu.)
echo.
%PYEXE% -u scripts\pipeline\run_one_model.py %MID%
echo.
pause >nul
goto :menu

:tao_bao_cao
call :tim_python
if "%PYEXE%"=="" pause
if "%PYEXE%"=="" goto :menu
if not exist "results" goto :chua_co_results
echo.
%PYEXE% scripts\tien_ich\build_html_report.py --results-dir results --out report.html
if errorlevel 1 goto :bao_cao_loi
echo.
echo Da tao report.html. Dang mo bang trinh duyet mac dinh ...
start "" "report.html"
pause >nul
goto :menu

:chua_co_results
echo.
echo [LOI] Chua co thu muc results\. Hay chay it nhat mot buoc truoc
echo ^(chon 1, 2 hoac 3 o menu^).
pause >nul
goto :menu

:bao_cao_loi
echo [LOI] Tao bao cao that bai. Xem thong bao loi o tren.
pause >nul
goto :menu

:ketthuc
echo.
echo Tam biet.
exit /b
