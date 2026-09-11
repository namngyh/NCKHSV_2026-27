@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "JOBS=12"
title Mo phong khoi phuc tham so -- MS-GARCH-X / EGARCH-X

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
echo Hay chay cai_dat_va_chay_full.bat truoc it nhat mot lan.
goto :eof

:menu
cls
echo ================================================================
echo  MO PHONG KHOI PHUC THAM SO (parameter recovery)
echo ================================================================
echo.
echo  PR-A  MS-GARCH-X Student-t, 2 trang thai   S1,S2,S3   B=100, n=2000
echo  PR-B  EGARCH-X Student-t,   1 trang thai   E1,E2,E3   B=100, n=2000
echo.
echo  [1] CHAY TAT CA -- PR-A xong thi TU DONG sang PR-B   (khuyen nghi)
echo  [2] Chi chay PR-A (MS)
echo  [3] Chi chay PR-B (EGARCH)
echo  [4] Xem tien do (cap nhat lien tuc, Ctrl+C de thoat)
echo  [5] Gop ket qua ca hai (khong tinh toan lai)
echo  [6] Chay thu nhanh (~4 phut) -- kiem tra may truoc khi chay that
echo  [7] Doi so tien trinh song song (dang dat: %JOBS%)
echo  [0] Thoat
echo.
echo  Ca hai deu BO QUA phan da co checkpoint, nen dung giua chung roi chay
echo  lai khong mat gi. Dung mo hai lan chay cung luc cho cung mot muc.
echo.
set "CHON="
set /p "CHON=Chon muc: "
rem Neu khong doc duoc gi (vd. chay khong co ban phim, stdin dong) thi THOAT
rem chu khong quay lai menu: quay lai se thanh vong lap vo han chay het toc
rem do, an mat mot nhan CPU cua chinh lan mo phong.
if not defined CHON goto :het
if "%CHON%"=="" goto :het
if "%CHON%"=="1" goto :chay_tat_ca
if "%CHON%"=="2" goto :chay_pr_a
if "%CHON%"=="3" goto :chay_pr_b
if "%CHON%"=="4" goto :tien_do
if "%CHON%"=="5" goto :gop
if "%CHON%"=="6" goto :chay_thu
if "%CHON%"=="7" goto :doi_jobs
if "%CHON%"=="0" goto :het
goto :menu

:chay_tat_ca
call :tim_python
if not defined PYEXE goto :xong_mot_muc
cls
echo ================================================================
echo  CHAY TAT CA: PR-A (MS) roi TU DONG sang PR-B (EGARCH)
echo  %JOBS% tien trinh song song. Ctrl+C de dung bat cu luc nao.
echo ================================================================
echo.
echo [Giai doan 1/2] PR-A -- MS-GARCH-X Student-t, S1+S2+S3, B=100
echo.
call :goi_pr_a
echo.
echo ================================================================
echo [Giai doan 1/2] PR-A da ket thuc luc %TIME%
echo.
echo Mot so lan lap that bai (NaN) la BINH THUONG -- vd. seed 28 cua S1
echo phan ky o ca 3 diem khoi tao. Bang gop se ghi "THIEU (99/100)".
echo ================================================================
echo.
echo [Giai doan 2/2] PR-B -- EGARCH-X Student-t, E1+E2+E3, B=100
echo.
call :goi_pr_b
echo.
echo ================================================================
echo  XONG CA HAI luc %TIME%. Dung muc [4] de xem bang tong ket.
echo ================================================================
goto :xong_mot_muc

:chay_pr_a
call :tim_python
if not defined PYEXE goto :xong_mot_muc
cls
echo === PR-A: MS-GARCH-X Student-t, S1+S2+S3, B=100, %JOBS% tien trinh ===
echo.
call :goi_pr_a
goto :xong_mot_muc

:chay_pr_b
call :tim_python
if not defined PYEXE goto :xong_mot_muc
cls
echo === PR-B: EGARCH-X Student-t, E1+E2+E3, B=100, %JOBS% tien trinh ===
echo.
call :goi_pr_b
goto :xong_mot_muc

rem --- Hai thu tuc goi that su, de muc [1] va muc [2]/[3] dung CHUNG mot lenh,
rem --- khong bi lech tham so khi sau nay sua mot cho ma quen cho kia.
:goi_pr_a
%PYEXE% scripts\chay_mo_phong_song_song.py --scenarios S1,S2,S3 --methods mle,vb_mean_field,vb_full_rank --B 100 --jobs %JOBS%
goto :eof

:goi_pr_b
%PYEXE% scripts\chay_mo_phong_egarch.py --jobs %JOBS%
goto :eof

:tien_do
call :tim_python
if not defined PYEXE goto :xong_mot_muc
cls
%PYEXE% scripts\tien_do_mo_phong.py --lap --moi 60 --jobs %JOBS%
goto :xong_mot_muc

:gop
call :tim_python
if not defined PYEXE goto :xong_mot_muc
cls
echo === Gop PR-A -^> 07_simulation_S1.pt, S2, S3 ===
%PYEXE% scripts\chay_mo_phong_song_song.py --combine --scenarios S1,S2,S3 --B 100
echo.
echo === Gop PR-B -^> mo_phong_egarch\E1.pt, E2.pt, E3.pt ===
%PYEXE% scripts\chay_mo_phong_egarch.py --combine
goto :xong_mot_muc

:chay_thu
call :tim_python
if not defined PYEXE goto :xong_mot_muc
cls
echo === Chay thu: cau hinh nho, ghi vao thu muc RIENG, khong dung ket qua that ===
echo.
echo --- PR-A ---
%PYEXE% scripts\chay_mo_phong_song_song.py --preset dry-run --scenarios S1 --methods mle,vb_mean_field,vb_full_rank
echo.
echo --- PR-B ---
%PYEXE% scripts\chay_mo_phong_egarch.py --nhanh --jobs %JOBS%
goto :xong_mot_muc

:doi_jobs
echo.
echo May nay co 8 nhan vat ly / 16 luong. Khuyen nghi 12.
echo Dat thap hon neu ban con muon dung may cho viec khac.
set /p "JOBS=So tien trinh song song: "
goto :menu

:xong_mot_muc
echo.
pause
goto :menu

:het
endlocal
