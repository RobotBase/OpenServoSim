@echo off
echo =======================================================
echo OpenServoSim - Installation and Quick Start Wrapper
echo =======================================================
echo.

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [2/3] Installing required dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install pip dependencies.
    pause
    exit /b 1
)

echo.
echo [3/3] Ready! Launching the high-speed UVC walking demo...
echo.
echo Tip: While walking, press the 'P' key in the MuJoCo viewer 
echo to apply a lateral push and watch the UVC algorithm recover!
echo.
pause

python examples/05_uvc_walk.py --speed fast
