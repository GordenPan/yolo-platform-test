@echo off
title YOLO Platform Installer
cd /d "%~dp0"

echo ============================================
echo   YOLO Platform - Installer
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('python --version') do echo Found %%v
echo.

echo [1/3] Upgrading pip ...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing dependencies from requirements.txt ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See messages above.
    pause
    exit /b 1
)

echo.
echo [3/3] Checking GPU / PyTorch ...
echo.

REM Detect an NVIDIA GPU
set "HAS_GPU="
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "HAS_GPU=1"

REM Check whether the installed torch can use CUDA
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 (
    echo [OK] PyTorch with CUDA is available - training will use the GPU.
    goto done
)

if defined HAS_GPU (
    echo [NOTE] An NVIDIA GPU was detected, but the installed PyTorch is CPU-only.
    echo        Training will run on CPU ^(slow^). To enable GPU acceleration,
    echo        install a CUDA build of PyTorch, for example ^(CUDA 11.8^):
    echo.
    echo        python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
    echo.
    echo        Pick the URL matching your CUDA version from https://pytorch.org/get-started/locally/
) else (
    echo [NOTE] No NVIDIA GPU detected - training will run on CPU.
)

:done
echo.
echo ============================================
echo   Installation finished.
echo   Start the platform by double-clicking start.bat
echo ============================================
pause
exit /b 0
