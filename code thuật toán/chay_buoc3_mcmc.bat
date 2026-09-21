@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
title Buoc 3 -- MCMC tren mo hinh thang cuoc

set "PYEXE="
py -c "import torch" >nul 2>nul
if not errorlevel 1 set "PYEXE=py"
if not defined PYEXE python -c "import torch" >nul 2>nul
if not defined PYEXE if not errorlevel 1 set "PYEXE=python"
if not defined PYEXE goto :khong_co_python
cls
echo ================================================================
echo  BUOC 3 -- MCMC (NUTS) tren mo hinh thang cuoc cua Buoc 1
echo ================================================================
echo.
echo  Mo hinh thang cuoc: EGARCH-X Student-t (8 tham so, MOT trang thai)
echo  Thang ca bon tieu chi: AIC 5606,87 - BIC 5650,54
echo                         QLIKE 0,9234 - NLS 1,5203
echo.
echo  Cau hinh: 4 chuoi x (1000 warmup + 400 mau), do sau cay toi da 7.
echo  Bon chuoi chay TUAN TU, khong song song.
echo.
echo  DA SUA so voi lan chay truoc:
echo    Lan truoc khong truyen diem khoi tao nen Pyro tu khoi tao ngau
echo    nhien. Chuoi dau chay tot (buoc tich phan 0,70 - 3,7 giay/vong)
echo    nhung mot chuoi sau roi vao vung theta = +1,03 (NGUOC dau don
echo    bay), beta = 0,32, roi buoc tich phan sup xuong 1,8e-07 va chuoi
echo    dung yen o 43 giay/vong.
echo    Nay bon chuoi khoi tao quanh nghiem MLE, phan tan 2 SAI SO CHUAN
echo    cua tung tham so. Phan tan de R-hat con y nghia chan doan, nhung
echo    khong du de nem chuoi sang che do khac.
echo.
echo  QUAN TRONG: neu lan chay MCMC cu VAN DANG CHAY thi dong no truoc
echo  (Ctrl+C trong cua so do), khong se hai tien trinh tranh CPU.
echo.
echo  Tien do hien truc tiep tren cua so nay, dang thanh tqdm:
echo     Warmup:  12%%^|#         ^| 168/1400 [05:23^<39:12, 1.91s/it, step size=7.03e-01, acc. prob=0.891]
echo  Ba con so dang theo doi:
echo     s/it       - nhip; lan truoc tot la ~3,7, hong la ^>40
echo     step size  - buoc tich phan; tot la ^>0,01, hong la ~1e-07
echo     acc. prob  - nen lang ve khoang 0,8
echo.
pause

echo.
%PYEXE% scripts\pipeline\pipeline_all_models.py

echo.
echo ================================================================
echo  KET THUC luc %TIME%
echo  Dong chan doan "R-hat lon nhat / ESS nho nhat / phan ky" o tren.
echo  Nguong: R-hat ^< 1,01 va ESS ^> 400 thi coi la hoi tu.
echo ================================================================
pause
goto :het

:khong_co_python
echo [LOI] Khong tim thay Python trong PATH.
pause

:het
endlocal
