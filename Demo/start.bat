@echo off
chcp 65001 >nul
title AI-PoisonGuard

echo ========================================
echo   AI-PoisonGuard Platform Startup
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

echo [1/3] Building frontend...
cd frontend
call npm run build
cd ..

echo [2/3] Deploying static files...
if exist "backend\static" rmdir /s /q "backend\static"
xcopy /e /i /q "frontend\dist" "backend\static"

echo [3/3] Starting backend server...
echo.
echo   Platform: http://localhost:8000
echo   API docs: http://localhost:8000/docs
echo   Press Ctrl+C to stop
echo.
cd backend
..\venv\Scripts\python.exe -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
pause
