@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Tien do MS-GARCH-X
python scripts	ien_do.py --lap --moi 30
pause
