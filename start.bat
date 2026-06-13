@echo off
title YOLO Platform Launcher
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM Skip backend startup if it is already running
curl -s -o nul -m 2 http://127.0.0.1:8000/api/health
if not errorlevel 1 (
    echo Backend is already running, skip.
    goto frontend
)

echo [1/2] Starting backend API on port 8000 ...
start "YOLO Platform - Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

REM Wait until the backend is ready (max 60s)
set /a tries=0
:wait_backend
set /a tries+=1
if %tries% gtr 60 (
    echo [WARN] Backend not ready within 60s, opening frontend anyway.
    echo        Please check the Backend window for errors.
    goto frontend
)
ping -n 2 127.0.0.1 >nul
curl -s -o nul -m 2 http://127.0.0.1:8000/api/health
if errorlevel 1 goto wait_backend
echo Backend is ready.

:frontend
echo [2/2] Starting frontend UI on port 8501 (browser opens automatically) ...
start "YOLO Platform - Frontend" cmd /k "cd /d "%~dp0" && python -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1"

echo.
echo Done:
echo   Web UI    http://localhost:8501
echo   API docs  http://localhost:8000/docs
echo.
echo To stop the platform, just close the Backend and Frontend windows.
ping -n 6 127.0.0.1 >nul
exit /b 0
