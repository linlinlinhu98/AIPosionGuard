@echo off
chcp 65001 >nul
title AI-PoisonGuard Install

echo ========================================
echo   AI-PoisonGuard Installation
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Creating Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Created venv
) else (
    echo   venv already exists
)

echo [2/3] Installing Python dependencies...
venv\Scripts\python.exe -m pip install -r backend\requirements.txt -q
echo   Done

echo [3/3] Installing frontend dependencies...
cd frontend
call npm install
cd ..
echo   Done

echo.
echo ========================================
echo   Installation complete!
echo   Run start.bat to launch the platform.
echo ========================================
pause
